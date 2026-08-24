"""Tests for Arabic calligraphy OCR and style recognition (#234).

Fully offline: the knowledge base is bundled JSON, the engine under test is
the deterministic marker-driven stub, and endpoint tests run against the ASGI
transport with the stub provider — no network access is ever made. A dummy
GEMINI_API_KEY is provided only so ``import main`` satisfies Settings; CI has
no real secrets.
"""

from httpx import ASGITransport, AsyncClient
import pytest

from calligraphy_ocr import (
    CalligraphyAnalysis,
    CalligraphyStyleCatalog,
    StubCalligraphyEngine,
    calibrate,
    sniff_image_mime,
    style_catalog,
    to_manuscript_payload,
)

# Minimal valid image headers fabricated in-test (no real image data needed).
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8\xff\xe0"


def png_image(*markers: bytes) -> bytes:
    return PNG_MAGIC + b"".join(markers) + b"\x00\x00fake-pixel-data"


def jpeg_image(*markers: bytes) -> bytes:
    return JPEG_MAGIC + b"".join(markers) + b"\x00\x00fake-pixel-data"


def make_stub_client(monkeypatch):
    """ASGI client wired to the app with the stub calligraphy provider."""
    import os

    os.environ.setdefault("GEMINI_API_KEY", "test-key")
    import main

    monkeypatch.setattr(main.settings, "calligraphy_provider", "stub")
    return AsyncClient(transport=ASGITransport(app=main.app), base_url="http://test")


# ---------------------------------------------------------------------------
# Style knowledge base
# ---------------------------------------------------------------------------


class TestStyleCatalog:
    def test_loads_all_core_styles(self):
        assert set(style_catalog.styles) == {"naskh", "thuluth", "diwani", "kufi", "ruqah", "modern"}

    def test_required_fields_present(self):
        required = {"label", "name", "arabic_name", "period_origin", "decorativeness"}
        for style in style_catalog.styles.values():
            data = style.model_dump()
            missing = required - set(data)
            assert not missing, f"{style.label} missing {missing}"

    def test_decorativeness_within_bounds(self):
        for style in style_catalog.styles.values():
            assert 0.0 <= style.decorativeness <= 1.0

    def test_labels_are_unique(self):
        labels = [style.label for style in style_catalog.styles.values()]
        assert len(labels) == len(set(labels))

    def test_lookup_by_label(self):
        assert style_catalog.lookup("kufi") is not None
        assert style_catalog.lookup("KUFI").label == "kufi"

    def test_lookup_by_alias_spelling_variants(self):
        # Includes the orthographic variants from the issue: ruqʿah / rüqah.
        for variant in ("ruqah", "ruqʿah", "rüqah", "ruq'ah", "ruqaa"):
            resolved = style_catalog.lookup(variant)
            assert resolved is not None, f"alias {variant!r} did not resolve"
            assert resolved.label == "ruqah"

    def test_lookup_by_kufic_alias(self):
        assert style_catalog.lookup("kufic").label == "kufi"

    def test_lookup_unknown_returns_none(self):
        assert style_catalog.lookup("comic sans") is None

    def test_duplicate_labels_rejected(self, tmp_path):
        import json

        bad_path = tmp_path / "bad_kb.json"
        payload = {
            "styles": [
                {"label": "naskh", "name": "N", "arabic_name": "ن", "period_origin": "x", "decorativeness": 0.2},
                {"label": "naskh", "name": "N2", "arabic_name": "ن", "period_origin": "y", "decorativeness": 0.3},
            ]
        }
        bad_path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="duplicate"):
            CalligraphyStyleCatalog(bad_path)


# ---------------------------------------------------------------------------
# Magic-byte sniffing
# ---------------------------------------------------------------------------


class TestSniffImageMime:
    def test_jpeg(self):
        assert sniff_image_mime(jpeg_image()) == "image/jpeg"

    def test_png(self):
        assert sniff_image_mime(png_image()) == "image/png"

    def test_unsupported(self):
        assert sniff_image_mime(b"GIF89a" + b"\x00" * 16) is None
        assert sniff_image_mime(b"") is None
        assert sniff_image_mime(b"not an image at all") is None


# ---------------------------------------------------------------------------
# Stub engine (deterministic, marker-driven)
# ---------------------------------------------------------------------------


class TestStubEngine:
    def test_default_success_png(self):
        analysis = StubCalligraphyEngine().analyze(png_image(), "image/png")
        assert isinstance(analysis, CalligraphyAnalysis)
        assert analysis.extracted_text.strip()
        assert analysis.style.label == "naskh"
        assert analysis.regions
        assert analysis.overall_confidence > 0.5

    def test_marker_selects_style(self):
        analysis = StubCalligraphyEngine().analyze(png_image(b"style=kufi"), "image/png")
        assert analysis.style.label == "kufi"
        assert analysis.metadata.engine == "stub"

    def test_deterministic(self):
        engine = StubCalligraphyEngine()
        a = engine.analyze(png_image(b"[heavy]"), "image/png")
        b = engine.analyze(png_image(b"[heavy]"), "image/png")
        assert a == b

    @pytest.mark.parametrize("magic", [JPEG_MAGIC, PNG_MAGIC])
    def test_accepts_both_headers(self, magic):
        analysis = StubCalligraphyEngine().analyze(magic + b"payload", "image/jpeg")
        assert analysis.extracted_text


# ---------------------------------------------------------------------------
# Confidence calibration & warnings
# ---------------------------------------------------------------------------


class TestCalibration:
    def test_low_confidence_produces_warnings(self):
        analysis = StubCalligraphyEngine().analyze(png_image(b"[lowconf]"), "image/png")
        assert analysis.overall_confidence < 0.35
        assert analysis.warnings, "low-confidence run must emit warnings"
        assert any("Low-confidence region" in w for w in analysis.warnings)
        assert any("uncertain" in w for w in analysis.warnings)

    def test_heavy_decoration_warning_for_ornate_style(self):
        analysis = StubCalligraphyEngine().analyze(png_image(b"style=diwani", b"[heavy]"), "image/png")
        assert analysis.decorations_detected is True
        assert any("Heavy decoration" in w for w in analysis.warnings)

    def test_clean_run_has_no_warnings(self):
        analysis = StubCalligraphyEngine().analyze(png_image(), "image/png")
        assert analysis.overall_confidence >= 0.35
        assert analysis.warnings == []

    def test_no_legible_text_warning(self):
        analysis = StubCalligraphyEngine().analyze(png_image(b"[nolegible]"), "image/png")
        assert analysis.extracted_text == ""
        assert not analysis.regions
        # No regions means only the (weak) style signal feeds the blend.
        assert analysis.overall_confidence < 0.5
        assert any("No legible" in w for w in analysis.warnings)

    def test_overall_is_blend_of_region_style_completeness(self):
        analysis = StubCalligraphyEngine().analyze(png_image(), "image/png")
        region_mean = sum(r.confidence for r in analysis.regions) / len(analysis.regions)
        expected = 0.6 * region_mean + 0.25 * analysis.style.confidence + 0.15 * 1.0
        assert abs(analysis.overall_confidence - expected) < 0.01

    def test_calibrate_clamps_to_unit_interval(self):
        analysis = StubCalligraphyEngine().analyze(png_image(), "image/png")
        analysis.overall_confidence = -5.0
        result = calibrate(analysis)
        assert 0.0 <= result.overall_confidence <= 1.0


# ---------------------------------------------------------------------------
# Response model round-trip + manuscript payload mapping
# ---------------------------------------------------------------------------


class TestResponseModelAndPayload:
    def test_model_dump_validate_round_trip(self):
        analysis = StubCalligraphyEngine().analyze(png_image(b"style=diwani"), "image/png")
        revived = CalligraphyAnalysis.model_validate(analysis.model_dump())
        assert revived == analysis

    def test_json_round_trip(self):
        analysis = StubCalligraphyEngine().analyze(jpeg_image(), "image/jpeg")
        revived = CalligraphyAnalysis.model_validate_json(analysis.model_dump_json())
        assert revived == analysis

    def test_manuscript_payload_field_mapping(self):
        analysis = StubCalligraphyEngine().analyze(png_image(b"style=thuluth"), "image/png")
        payload = to_manuscript_payload(analysis)

        assert payload["text"]["raw"] == analysis.extracted_text
        assert payload["text"]["normalized"] == analysis.transcription_normalized
        assert payload["regions"][0]["text"] == analysis.regions[0].text
        assert payload["regions"][0]["confidence"] == analysis.regions[0].confidence
        assert payload["regions"][0]["bbox"] == analysis.regions[0].bbox_hint
        assert payload["script"]["family"] == analysis.style.label
        assert payload["script"]["variants"] == analysis.style.alternates
        assert payload["script"]["classification_confidence"] == analysis.style.confidence
        assert payload["script"]["classical"] == analysis.period.classical
        assert payload["script"]["period_estimate"] == analysis.period.era
        assert payload["quality"]["decorated"] == analysis.decorations_detected
        assert payload["quality"]["legibility"] == analysis.legibility
        assert payload["quality"]["confidence"] == analysis.overall_confidence
        assert payload["quality"]["warnings"] == analysis.warnings
        assert payload["source"]["engine"] == "stub"


# ---------------------------------------------------------------------------
# Endpoint contract (offline, stub provider)
# ---------------------------------------------------------------------------


class TestCalligraphyEndpoint:
    @pytest.mark.asyncio
    async def test_success_returns_analysis(self, monkeypatch):
        async with make_stub_client(monkeypatch) as client:
            res = await client.post(
                "/calligraphy/analyze",
                files={"file": ("art.png", png_image(b"style=diwani"), "image/png")},
            )
        assert res.status_code == 200
        body = res.json()
        assert body["extracted_text"].strip()
        assert body["style"]["label"] == "diwani"
        assert isinstance(body["warnings"], list)
        assert body["metadata"]["engine"] == "stub"

    @pytest.mark.asyncio
    async def test_wrong_magic_bytes_rejected_415(self, monkeypatch):
        async with make_stub_client(monkeypatch) as client:
            res = await client.post(
                "/calligraphy/analyze",
                files={"file": ("art.gif", b"GIF89a" + b"\x00" * 32, "image/gif")},
            )
        assert res.status_code == 415

    @pytest.mark.asyncio
    async def test_oversize_rejected_413(self, monkeypatch):
        import main

        monkeypatch.setattr(main.settings, "calligraphy_max_image_bytes", 64)
        async with make_stub_client(monkeypatch) as client:
            res = await client.post(
                "/calligraphy/analyze",
                files={"file": ("big.png", png_image() + b"\x00" * 256, "image/png")},
            )
        assert res.status_code == 413

    @pytest.mark.asyncio
    async def test_no_legible_calligraphy_422(self, monkeypatch):
        async with make_stub_client(monkeypatch) as client:
            res = await client.post(
                "/calligraphy/analyze",
                files={"file": ("blank.png", png_image(b"[nolegible]"), "image/png")},
            )
        assert res.status_code == 422
        assert "No legible" in res.json()["detail"]

    @pytest.mark.asyncio
    async def test_stub_disabled_in_production_503(self, monkeypatch):
        monkeypatch.setenv("ENVIRONMENT", "production")
        async with make_stub_client(monkeypatch) as client:
            res = await client.post(
                "/calligraphy/analyze",
                files={"file": ("art.png", png_image(), "image/png")},
            )
        assert res.status_code == 503

    @pytest.mark.asyncio
    async def test_missing_file_rejected_422(self, monkeypatch):
        async with make_stub_client(monkeypatch) as client:
            res = await client.post("/calligraphy/analyze")
        assert res.status_code == 422
