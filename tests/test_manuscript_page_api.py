"""HTTP contract tests for manuscript OCR and page analysis (#854).

These drive the FastAPI routes with small synthetic requests. The OCR engine is
the module's local stub, and the pipeline entry points are wrapped in spies so
each test can assert that a rejected request never reached processing. Nothing
here calls an external provider, needs credentials, or uses a real scan.
"""

from __future__ import annotations

import base64
import io
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

import arabic_ocr
import page_analysis
from main import app
from media_upload_limits import MAX_IMAGE_BASE64_CHARS, MAX_IMAGE_BYTES

client = TestClient(app)

# Synthetic payloads: valid signatures, no real image content.
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
JPEG = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00\xff\xdb\x00C\x00"
WEBP = b"RIFF\x24\x00\x00\x00WEBPVP8 \x18\x00\x00\x00" + b"\x00" * 24
PDF = b"%PDF-1.4\n%synthetic test document\n"
TEXT = b"plain text, not an image"

PNG_B64 = base64.b64encode(PNG).decode()
BASE64_ROUTES = ["/arabic-ocr/process", "/arabic-ocr/detect-style", "/arabic-ocr/preprocess"]


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


@pytest.fixture
def pipeline_calls(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    """Record every call into OCR processing; tests assert it stays empty on rejection."""
    calls: list[str] = []

    real_process = arabic_ocr.process_manuscript_page
    real_detect = arabic_ocr.detect_calligraphy_style
    real_preprocess = arabic_ocr.preprocess_manuscript_image

    async def process_spy(*args: Any, **kwargs: Any) -> Any:
        calls.append("process")
        return await real_process(*args, **kwargs)

    def detect_spy(*args: Any, **kwargs: Any) -> Any:
        calls.append("detect")
        return real_detect(*args, **kwargs)

    def preprocess_spy(*args: Any, **kwargs: Any) -> Any:
        calls.append("preprocess")
        return real_preprocess(*args, **kwargs)

    monkeypatch.setattr(arabic_ocr, "process_manuscript_page", process_spy)
    monkeypatch.setattr(arabic_ocr, "detect_calligraphy_style", detect_spy)
    monkeypatch.setattr(arabic_ocr, "preprocess_manuscript_image", preprocess_spy)
    yield calls


@pytest.fixture
def analysis_calls(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[int]]:
    calls: list[int] = []
    real = page_analysis.analyze_page

    def spy(page: page_analysis.PageInput) -> page_analysis.PageAnalysisResponse:
        calls.append(len(page.blocks))
        return real(page)

    monkeypatch.setattr(page_analysis, "analyze_page", spy)
    yield calls


def _page(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "width": 1000,
        "height": 1500,
        "rtl": False,
        "blocks": [
            {"x": 300, "y": 20, "width": 400, "height": 60, "text": "Chapter on Patience", "font_size": 24},
            {
                "x": 100,
                "y": 300,
                "width": 800,
                "height": 400,
                "text": "Be patient, as in Surah 2:153, and see Bukhari #1. Surah 1:200 is a misprint.",
                "font_size": 12,
                "confidence": 0.91,
            },
        ],
        "consent_to_process": True,
    }
    body.update(overrides)
    return body


# ---------------------------------------------------------------------------
# /page-analysis/analyze
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("consent", [None, False])
def test_page_analysis_without_affirmative_consent_is_400_and_not_analyzed(
    consent: bool | None, analysis_calls: list[int]
) -> None:
    body = _page()
    if consent is None:
        body.pop("consent_to_process")
    else:
        body["consent_to_process"] = consent
    response = client.post("/page-analysis/analyze", json=body)
    assert response.status_code == 400
    assert "consent" in response.json()["detail"]
    assert analysis_calls == []


def test_page_analysis_success_shape_and_provenance(analysis_calls: list[int]) -> None:
    body = _page()
    response = client.post("/page-analysis/analyze", json=body)
    assert response.status_code == 200
    data = response.json()
    assert analysis_calls == [2]
    assert data["contract_version"] == page_analysis.CONTRACT_VERSION

    source = body["blocks"][1]["text"]
    by_ref = {ref["reference"]: ref for ref in data["references"]}
    quran = by_ref["2:153"]
    hadith = by_ref["Sahih al-Bukhari"]
    for ref in (quran, hadith):
        assert ref["block_index"] == 1
        assert source[ref["start"] : ref["end"]] == ref["text"]
        assert ref["status"] == "validated"
        assert ref["ocr_confidence"] == 0.91
        assert ref["ocr_confidence_band"] == "high"
        assert ref["requires_review"] is False
    assert hadith["hadith_number"] == 1

    [rejected] = data["rejected_references"]
    assert rejected["text"] == "Surah 1:200"
    assert any("rejected" in warning for warning in data["warnings"])


def test_page_analysis_without_ocr_confidence_reports_unavailable() -> None:
    body = _page()
    del body["blocks"][1]["confidence"]
    data = client.post("/page-analysis/analyze", json=body).json()
    assert {ref["ocr_confidence_band"] for ref in data["references"]} == {"unavailable"}
    assert all(ref["ocr_confidence"] is None and ref["requires_review"] for ref in data["references"])


@pytest.mark.parametrize(
    "overrides",
    [
        {"width": 25000},
        {"blocks": [{"x": 0, "y": 0, "width": 1, "height": 1, "text": "x" * 20001}]},
        {"blocks": [{"x": 0, "y": 0, "width": 1, "height": 1, "text": "x"}] * 5001},
        {"blocks": [{"x": 0, "y": 0, "width": 1, "height": 1, "text": "x" * 20000}] * 11},
        {"blocks": [{"x": -5, "y": 0, "width": 1, "height": 1, "text": "x"}]},
        {"blocks": [{"x": 0, "y": 0, "width": 1, "height": 1, "text": "x", "confidence": 2}]},
    ],
    ids=["dimension", "block-text", "block-count", "page-text", "negative-x", "confidence-range"],
)
def test_page_analysis_limit_violations_are_422_and_not_analyzed(
    overrides: dict[str, Any], analysis_calls: list[int]
) -> None:
    response = client.post("/page-analysis/analyze", json=_page(**overrides))
    assert response.status_code == 422
    assert analysis_calls == []


def test_page_analysis_empty_page_succeeds_with_warning() -> None:
    data = client.post("/page-analysis/analyze", json=_page(blocks=[])).json()
    assert data["references"] == []
    assert any("No text" in warning for warning in data["warnings"])


# ---------------------------------------------------------------------------
# Base64 OCR routes: /process, /detect-style, /preprocess
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("route", BASE64_ROUTES)
@pytest.mark.parametrize("consent", [None, False])
def test_base64_routes_require_affirmative_consent(route: str, consent: bool | None, pipeline_calls: list[str]) -> None:
    body: dict[str, Any] = {"image_base64": PNG_B64}
    if consent is not None:
        body["consent_to_process"] = consent
    response = client.post(route, json=body)
    assert response.status_code == 400
    assert pipeline_calls == []


@pytest.mark.parametrize("route", BASE64_ROUTES)
def test_base64_routes_accept_consented_png(route: str, pipeline_calls: list[str]) -> None:
    response = client.post(route, json={"image_base64": PNG_B64, "consent_to_process": True})
    assert response.status_code == 200
    assert pipeline_calls


@pytest.mark.parametrize("route", BASE64_ROUTES)
@pytest.mark.parametrize(
    ("image_base64", "status"),
    [
        ("not-valid-base64!!!", 422),
        ("data:text/html;base64," + PNG_B64, 422),
        (_b64(PDF), 415),
        (_b64(TEXT), 415),
        ("", 400),
    ],
    ids=["malformed", "data-url-not-image", "pdf", "text", "empty"],
)
def test_base64_routes_reject_bad_payloads_before_processing(
    route: str, image_base64: str, status: int, pipeline_calls: list[str]
) -> None:
    response = client.post(route, json={"image_base64": image_base64, "consent_to_process": True})
    assert response.status_code == status
    assert pipeline_calls == []


@pytest.mark.parametrize(
    "image_base64",
    [
        "data:image/png;base64," + PNG_B64,
        "\n".join(PNG_B64[i : i + 76] for i in range(0, len(PNG_B64), 76)) + "\n",
        _b64(JPEG),
        _b64(WEBP),
    ],
    ids=["png-data-url", "line-wrapped", "jpeg", "webp"],
)
def test_process_accepts_supported_encodings_and_formats(image_base64: str, pipeline_calls: list[str]) -> None:
    response = client.post("/arabic-ocr/process", json={"image_base64": image_base64, "consent_to_process": True})
    assert response.status_code == 200
    assert pipeline_calls[0] == "process"


def test_process_rejects_oversized_encoded_image_with_413(pipeline_calls: list[str]) -> None:
    oversized = "A" * (MAX_IMAGE_BASE64_CHARS + 4)
    response = client.post("/arabic-ocr/process", json={"image_base64": oversized, "consent_to_process": True})
    assert response.status_code == 413
    assert pipeline_calls == []


def test_detect_style_bounds_text_sample() -> None:
    response = client.post(
        "/arabic-ocr/detect-style",
        json={"image_base64": PNG_B64, "consent_to_process": True, "text_sample": "x" * 20001},
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Multipart OCR route: /process-file
# ---------------------------------------------------------------------------


def _upload(name: str, data: bytes, content_type: str = "image/png") -> dict[str, Any]:
    return {"file": (name, io.BytesIO(data), content_type)}


def test_process_file_requires_affirmative_consent(pipeline_calls: list[str]) -> None:
    assert client.post("/arabic-ocr/process-file", files=_upload("page.png", PNG)).status_code == 400
    response = client.post(
        "/arabic-ocr/process-file", files=_upload("page.png", PNG), data={"consent_to_process": "false"}
    )
    assert response.status_code == 400
    assert pipeline_calls == []


@pytest.mark.parametrize(
    ("name", "data", "status"),
    [
        ("page.png", PDF, 415),
        ("page.pdf", PDF, 415),
        ("page.png", TEXT, 415),
        ("page.png", b"", 400),
        ("page.png", PNG[:8] + b"\x00" * MAX_IMAGE_BYTES, 413),
    ],
    ids=["pdf-named-png", "pdf", "text", "empty", "oversized"],
)
def test_process_file_rejects_by_content_not_name(
    name: str, data: bytes, status: int, pipeline_calls: list[str]
) -> None:
    response = client.post("/arabic-ocr/process-file", files=_upload(name, data), data={"consent_to_process": "true"})
    assert response.status_code == status
    assert pipeline_calls == []


def test_process_file_trusts_signature_over_extension(pipeline_calls: list[str]) -> None:
    response = client.post(
        "/arabic-ocr/process-file",
        files=_upload("page.png", JPEG, "image/png"),
        data={"consent_to_process": "true"},
    )
    assert response.status_code == 200
    assert pipeline_calls[0] == "process"
