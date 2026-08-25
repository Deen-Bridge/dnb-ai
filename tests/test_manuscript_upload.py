"""Offline tests for the manuscript upload endpoint (#233).

No secrets and no network: the provider is forced to the deterministic stub
engine, and every "image" is a fabricated byte header built in-test — no
committed binaries. The FastAPI app is exercised through httpx's ASGI
transport with Gemini untouched, matching the existing endpoint tests.
"""

import os
from types import SimpleNamespace
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

import manuscript_ocr
from manuscript_ocr import (
    GeminiManuscriptEngine,
    PillowPreprocessor,
    PoorQualityError,
    StubManuscriptEngine,
    UnsupportedFormatError,
    UploadTooLargeError,
    analysis_from_payload,
    create_manuscript_engine,
    validate_upload,
)

JPEG_MAGIC = b"\xff\xd8\xff"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
PDF_MAGIC = b"%PDF-1.7"


def _jpeg(marker: bytes = b"MSS-TYPE:quran") -> bytes:
    return JPEG_MAGIC + marker + b"\x00" * 48


def _fake_settings(**overrides: Any) -> SimpleNamespace:
    values: dict[str, Any] = dict(
        manuscripts_provider="stub",
        manuscripts_max_upload_bytes=1_048_576,
        manuscripts_min_confidence=0.35,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.fixture()
def client(monkeypatch):
    """ASGI client with the pipeline pinned to the offline stub provider."""
    os.environ.setdefault("GEMINI_API_KEY", "offline-test-key")
    import main

    monkeypatch.setattr(manuscript_ocr, "get_settings", lambda: _fake_settings())
    manuscript_ocr.manuscript_rate_limiter.reset()

    transport = ASGITransport(app=main.app)
    return AsyncClient(transport=transport, base_url="http://test")


# ---------------------------------------------------------------------------
# Upload validation (magic bytes + extension agreement)
# ---------------------------------------------------------------------------


class TestUploadValidation:
    @pytest.mark.parametrize(
        "data, filename, mime",
        [
            (_jpeg(), "scan.jpg", "image/jpeg"),
            (_jpeg(b"MSS-TYPE:hadith"), "scan.jpeg", "image/jpeg"),
            (PNG_MAGIC + b"\x00" * 48, "page.png", "image/png"),
            (PDF_MAGIC + b"\n%fake body", "manuscript.pdf", "application/pdf"),
        ],
    )
    def test_accepts_matching_magic_and_extension(self, data, filename, mime):
        upload = validate_upload(filename, data)
        assert upload.mime == mime
        assert upload.data == data

    def test_rejects_magic_extension_mismatch(self):
        with pytest.raises(UnsupportedFormatError, match="does not match"):
            validate_upload("photo.jpg", PNG_MAGIC + b"\x00" * 48)

    def test_rejects_unrecognized_magic(self):
        with pytest.raises(UnsupportedFormatError, match="Unsupported file"):
            validate_upload("drawing.png", b"GIF89a" + b"\x00" * 48)

    def test_rejects_disallowed_extension_even_with_valid_magic(self):
        """Valid JPEG bytes renamed .txt must not slip through on magic alone."""
        with pytest.raises(UnsupportedFormatError, match="extension"):
            validate_upload("notes.txt", _jpeg())

    def test_rejects_missing_extension(self):
        with pytest.raises(UnsupportedFormatError, match="extension"):
            validate_upload("", _jpeg())

    def test_rejects_empty_upload(self):
        with pytest.raises(UnsupportedFormatError):
            validate_upload("scan.jpg", b"")

    def test_size_cap_raises_before_format_checks(self):
        with pytest.raises(UploadTooLargeError, match="maximum"):
            validate_upload("scan.jpg", _jpeg(), max_bytes=8)


# ---------------------------------------------------------------------------
# Preprocessing (graceful degradation, PDF flagging)
# ---------------------------------------------------------------------------


class TestPreprocessing:
    def test_pdf_is_flagged_for_the_vision_backend_untouched(self):
        result = PillowPreprocessor().preprocess(PDF_MAGIC + b"\nblob", "application/pdf")
        assert result.needs_ocr_backend is True
        assert result.data == PDF_MAGIC + b"\nblob"
        assert result.mime == "application/pdf"
        assert result.warnings

    def test_garbage_body_falls_back_to_original_bytes_with_warning(self):
        """A valid header over junk content must degrade, never raise."""
        garbage = PNG_MAGIC + b"this is definitely not a real png payload"
        result = PillowPreprocessor().preprocess(garbage, "image/png")
        assert result.data == garbage
        assert result.warnings and "original image" in result.warnings[0]

    def test_valid_image_is_normalized_to_grayscale_png(self):
        pytest.importorskip("PIL")
        import io

        from PIL import Image

        buffer = io.BytesIO()
        Image.new("RGB", (40, 30), color=(200, 10, 10)).save(buffer, format="JPEG")
        jpeg_bytes = buffer.getvalue()
        result = PillowPreprocessor().preprocess(jpeg_bytes, "image/jpeg")
        assert result.mime == "image/png"
        with Image.open(io.BytesIO(result.data)) as normalized:
            assert normalized.mode == "L"

    def test_low_resolution_scan_is_upscaled(self):
        pytest.importorskip("PIL")
        import io

        from PIL import Image

        buffer = io.BytesIO()
        Image.new("L", (100, 80)).save(buffer, format="PNG")
        result = PillowPreprocessor().preprocess(buffer.getvalue(), "image/png")
        with Image.open(io.BytesIO(result.data)) as upscaled:
            assert upscaled.width >= PillowPreprocessor.LOW_RESOLUTION_WIDTH
        assert any("upscaled" in warning for warning in result.warnings)


# ---------------------------------------------------------------------------
# Stub engine determinism
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestStubEngine:
    async def test_happy_path_returns_structured_analysis(self):
        analysis = await StubManuscriptEngine().analyze(_jpeg(), "image/jpeg")
        assert analysis.extracted_text.strip()
        assert analysis.transliteration
        assert analysis.manuscript_type.label == "quran"
        assert 0.0 <= analysis.manuscript_type.confidence <= 1.0
        assert isinstance(analysis.printed, bool)
        assert analysis.quality.legibility >= 0.5
        assert analysis.confidence >= 0.5
        assert analysis.sections

    async def test_type_marker_selects_the_label(self):
        analysis = await StubManuscriptEngine().analyze(_jpeg(b"MSS-TYPE:hadith"), "image/jpeg")
        assert analysis.manuscript_type.label == "hadith"

    async def test_printed_marker_flips_the_flag(self):
        analysis = await StubManuscriptEngine().analyze(_jpeg(b"MSS-PRINTED"), "image/jpeg")
        assert analysis.printed is True

    async def test_low_quality_marker_yields_empty_low_confidence_output(self):
        analysis = await StubManuscriptEngine().analyze(_jpeg(b"MSS-LOW-QUALITY"), "image/jpeg")
        assert analysis.extracted_text == ""
        assert analysis.confidence < 0.35


# ---------------------------------------------------------------------------
# Payload normalization (loose model JSON -> typed response)
# ---------------------------------------------------------------------------


class TestPayloadNormalization:
    def test_clamps_confidences_and_unknown_labels(self):
        analysis = analysis_from_payload(
            {
                "extracted_text": "نص",
                "manuscript_type": {"label": "EPIC POEM", "confidence": 1.7},
                "quality": {"legibility": -3, "completeness": 2},
                "confidence": "not-a-number",
                "sections": [{"label": "header", "text": "نص", "confidence": 0.8}],
            }
        )
        assert analysis.manuscript_type.label == "unknown"
        assert analysis.manuscript_type.confidence == 1.0
        assert analysis.quality.legibility == 0.0
        assert analysis.quality.completeness == 1.0
        assert analysis.confidence == 0.0
        assert analysis.sections[0].label == "header"

    def test_non_dict_payload_raises_poor_quality(self):
        with pytest.raises(PoorQualityError):
            analysis_from_payload(["not", "a", "dict"])


# ---------------------------------------------------------------------------
# Provider switching
# ---------------------------------------------------------------------------


class TestProviderSwitching:
    def test_stub_provider_builds_stub_engine(self):
        assert isinstance(create_manuscript_engine("stub"), StubManuscriptEngine)

    def test_gemini_provider_builds_gemini_engine(self):
        engine = create_manuscript_engine("gemini")
        assert isinstance(engine, GeminiManuscriptEngine)
        assert engine.model_name == "gemini-2.5-flash"

    def test_unknown_provider_falls_back_to_gemini(self):
        assert isinstance(create_manuscript_engine("crystal-ball"), GeminiManuscriptEngine)

    def test_none_defers_to_configured_settings(self, monkeypatch):
        monkeypatch.setattr(manuscript_ocr, "get_settings", lambda: _fake_settings(manuscripts_provider="stub"))
        assert isinstance(create_manuscript_engine(None), StubManuscriptEngine)


# ---------------------------------------------------------------------------
# Pipeline quality gate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestPipelineQualityGate:
    async def test_empty_extraction_raises_poor_quality(self, monkeypatch):
        monkeypatch.setattr(manuscript_ocr, "get_settings", lambda: _fake_settings())
        with pytest.raises(PoorQualityError, match="No readable Arabic text"):
            await manuscript_ocr.analyze_manuscript_bytes("scan.jpg", _jpeg(b"MSS-LOW-QUALITY"))

    async def test_below_threshold_confidence_raises_poor_quality(self, monkeypatch):
        tight = _fake_settings(manuscripts_min_confidence=0.99)
        monkeypatch.setattr(manuscript_ocr, "get_settings", lambda: tight)
        with pytest.raises(PoorQualityError, match="below the accepted minimum"):
            await manuscript_ocr.analyze_manuscript_bytes("scan.jpg", _jpeg())

    async def test_preprocessing_warnings_are_merged_into_the_response(self, monkeypatch):
        monkeypatch.setattr(manuscript_ocr, "get_settings", lambda: _fake_settings())
        analysis = await manuscript_ocr.analyze_manuscript_bytes(
            "scan.pdf", PDF_MAGIC + b"\n%stub marker MSS-TYPE:tafsir body"
        )
        assert analysis.manuscript_type.label == "tafsir"
        assert any("PDF" in warning for warning in analysis.warnings)


# ---------------------------------------------------------------------------
# Endpoint integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestManuscriptEndpoint:
    async def test_happy_path_returns_structured_json(self, client):
        async with client:
            resp = await client.post(
                "/manuscripts/analyze",
                files={"file": ("page.jpg", _jpeg(b"MSS-TYPE:quran"), "image/jpeg")},
            )
            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert body["extracted_text"]
            assert body["transliteration"]
            assert body["manuscript_type"]["label"] == "quran"
            assert body["printed"] is False
            assert 0.0 <= body["confidence"] <= 1.0
            assert body["historical_context"]
            assert isinstance(body["warnings"], list)

    async def test_oversized_upload_is_413(self, client, monkeypatch):
        monkeypatch.setattr(
            manuscript_ocr,
            "get_settings",
            lambda: _fake_settings(manuscripts_max_upload_bytes=16),
        )
        async with client:
            resp = await client.post(
                "/manuscripts/analyze",
                files={"file": ("page.jpg", _jpeg(), "image/jpeg")},
            )
            assert resp.status_code == 413
            assert "maximum" in resp.json()["detail"]

    async def test_magic_extension_mismatch_is_415(self, client):
        async with client:
            resp = await client.post(
                "/manuscripts/analyze",
                files={"file": ("photo.jpg", PNG_MAGIC + b"\x00" * 48, "image/jpeg")},
            )
            assert resp.status_code == 415

    async def test_unrecognized_magic_is_415(self, client):
        async with client:
            resp = await client.post(
                "/manuscripts/analyze",
                files={"file": ("thing.png", b"GIF89a" + b"\x00" * 48, "image/png")},
            )
            assert resp.status_code == 415

    async def test_poor_extraction_is_422(self, client):
        async with client:
            resp = await client.post(
                "/manuscripts/analyze",
                files={"file": ("blur.jpg", _jpeg(b"MSS-LOW-QUALITY"), "image/jpeg")},
            )
            assert resp.status_code == 422
            assert "detail" in resp.json()

    async def test_pdf_upload_flows_through_with_backend_warning(self, client):
        async with client:
            resp = await client.post(
                "/manuscripts/analyze",
                files={
                    "file": (
                        "ms.pdf",
                        PDF_MAGIC + b"\n%MSS-TYPE:fiqh stub document",
                        "application/pdf",
                    )
                },
            )
            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert body["manuscript_type"]["label"] == "fiqh"
            assert any("PDF" in warning for warning in body["warnings"])

    async def test_rate_limit_blocks_a_flood(self, client, monkeypatch):
        monkeypatch.setattr(manuscript_ocr.manuscript_rate_limiter, "_max", 2)
        async with client:
            statuses = []
            for _ in range(4):
                resp = await client.post(
                    "/manuscripts/analyze",
                    files={"file": ("page.jpg", _jpeg(), "image/jpeg")},
                )
                statuses.append(resp.status_code)
            assert 429 in statuses

    async def test_missing_file_field_is_422(self, client):
        async with client:
            resp = await client.post("/manuscripts/analyze", files={"other": ("a.txt", b"x")})
            assert resp.status_code == 422
