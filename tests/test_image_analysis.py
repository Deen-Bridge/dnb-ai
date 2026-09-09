"""Offline tests for image content analysis (#135).

No secrets and no network: the translation provider is pinned to the
deterministic stub, validation runs against the bundled surah index, and the
app is exercised through httpx's ASGI transport. Fabricated byte headers
stand in for real images (matching the manuscript-upload test style).
"""

import os

import pytest
from httpx import ASGITransport, AsyncClient

os.environ.setdefault("GEMINI_API_KEY", "test-key")

import image_analysis  # noqa: E402
import main  # noqa: E402
from image_analysis import (  # noqa: E402
    ContentAnalysis,
    StubTranslationEngine,
    analyze_content,
    detect_hadith,
    extract_verses,
    normalize_arabic,
    ocr_cache,
)
from manuscript_ocr import (  # noqa: E402
    ManuscriptAnalysis,
    ManuscriptTypeClassification,
    QualityAssessment,
)

JPEG_MAGIC = b"\xff\xd8\xff"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
WEBP_MAGIC = b"RIFF\x00\x00\x00\x00WEBP"
PDF_MAGIC = b"%PDF-1.7"


@pytest.fixture()
def stub_provider(monkeypatch):
    """Pin the translation engine to the deterministic stub."""
    monkeypatch.setattr(image_analysis, "create_translation_engine", lambda provider=None: StubTranslationEngine())


@pytest.fixture()
async def client(monkeypatch):
    # Pin the manuscript OCR provider to the offline stub (settings are cached,
    # so monkeypatch the module-level get_settings used by the pipeline).
    from types import SimpleNamespace

    import manuscript_ocr

    monkeypatch.setattr(
        manuscript_ocr,
        "get_settings",
        lambda: SimpleNamespace(
            manuscripts_provider="stub",
            manuscripts_max_upload_bytes=10 * 1024 * 1024,
            manuscripts_min_confidence=0.35,
        ),
    )
    transport = ASGITransport(app=main.app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


def _analysis(text: str, label: str = "quran") -> ManuscriptAnalysis:
    return ManuscriptAnalysis(
        extracted_text=text,
        manuscript_type=ManuscriptTypeClassification(label=label, confidence=0.9),
        quality=QualityAssessment(legibility=0.9, completeness=0.9),
        confidence=0.9,
    )


# ---------------------------------------------------------------------------
# Canonical verse extraction
# ---------------------------------------------------------------------------


class TestVerseExtraction:
    def test_extracts_valid_reference(self):
        verses = extract_verses("Read 2:255, the Ayat al-Kursi.")
        assert len(verses) == 1
        verse = verses[0]
        assert verse.reference == "2:255"
        assert verse.in_bounds is True
        assert verse.surah_name == "Al-Baqarah"

    def test_rejects_out_of_bounds_reference(self):
        verses = extract_verses("This references 2:300 which does not exist.")
        assert len(verses) == 1
        assert verses[0].in_bounds is False

    def test_handles_mixed_valid_and_invalid(self):
        verses = extract_verses("2:255 is valid but 99:999 is not.")
        by_ref = {v.reference: v for v in verses}
        assert by_ref["2:255"].in_bounds is True
        assert by_ref["99:999"].in_bounds is False

    def test_surah_name_from_index(self):
        verses = extract_verses("See 112:1")
        assert verses[0].surah_name == "Al-Ikhlas"


class TestNormalizeArabic:
    def test_strips_diacritics(self):
        assert normalize_arabic("بِسْمِ اللَّهِ") == normalize_arabic("بسم الله")

    def test_folds_hamza_variants(self):
        assert normalize_arabic("أحمد") == normalize_arabic("احمد")


# ---------------------------------------------------------------------------
# Hadith detection
# ---------------------------------------------------------------------------


class TestHadithDetection:
    def test_detects_collections(self):
        found = detect_hadith("Reported in Sahih al-Bukhari and Sahih Muslim.")
        collections = {h.collection for h in found}
        assert "Sahih al-Bukhari" in collections
        assert "Sahih Muslim" in collections

    def test_no_false_positive(self):
        assert detect_hadith("A simple sentence with no collections.") == []


# ---------------------------------------------------------------------------
# Content analysis (translation pinned to stub)
# ---------------------------------------------------------------------------


class TestContentAnalysis:
    async def test_analyze_content_returns_analysis(self, stub_provider):
        analysis: ContentAnalysis = await analyze_content(_analysis("Quran 2:255 and Bukhari"))
        assert analysis.content_type == "quran"
        assert analysis.verses
        assert analysis.verses[0].reference == "2:255"
        assert analysis.hadith
        assert analysis.translation.translation
        assert analysis.validation["references_found"] >= 1

    async def test_analyze_content_skip_translation(self, stub_provider):
        analysis: ContentAnalysis = await analyze_content(_analysis("Surah 2:255"), skip_translation=True)
        assert analysis.translation.translation == ""

    async def test_content_type_from_hadith(self, stub_provider):
        analysis: ContentAnalysis = await analyze_content(_analysis("Narrated in Sahih al-Bukhari", label="hadith"))
        assert analysis.content_type == "hadith"


# ---------------------------------------------------------------------------
# Batch endpoint
# ---------------------------------------------------------------------------


class TestBatch:
    @pytest.fixture(autouse=True)
    def reset_cache(self):
        ocr_cache._entries.clear()
        yield

    async def test_batch_accepts_webp(self, client, monkeypatch):
        monkeypatch.setattr(
            image_analysis,
            "create_translation_engine",
            lambda provider=None: StubTranslationEngine(),
        )
        resp = await client.post(
            "/image-analysis/batch",
            files=[("files", ("page.webp", WEBP_MAGIC + b"MSS-TYPE:quran" + b"\x00" * 48, "image/webp"))],
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["succeeded"] == 1
        assert data["results"][0]["ok"] is True

    async def test_batch_isolates_failing_page(self, client, monkeypatch):
        monkeypatch.setattr(
            image_analysis,
            "create_translation_engine",
            lambda provider=None: StubTranslationEngine(),
        )
        good = ("good.jpg", JPEG_MAGIC + b"MSS-TYPE:quran" + b"\x00" * 48, "image/jpeg")
        bad = ("bad.txt", b"not-an-image", "text/plain")
        resp = await client.post(
            "/image-analysis/batch",
            files=[("files", good), ("files", bad)],
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert data["succeeded"] == 1
        assert data["failed"] == 1
        ok_results = [r for r in data["results"] if r["ok"]]
        bad_results = [r for r in data["results"] if not r["ok"]]
        assert len(ok_results) == 1
        assert len(bad_results) == 1
        assert bad_results[0]["filename"] == "bad.txt"

    async def test_batch_caches_identical_images(self, client, monkeypatch):
        monkeypatch.setattr(
            image_analysis,
            "create_translation_engine",
            lambda provider=None: StubTranslationEngine(),
        )
        payload = ("scan.png", PNG_MAGIC + b"MSS-TYPE:quran" + b"\x00" * 64, "image/png")
        first = await client.post("/image-analysis/batch", files=[("files", payload)])
        second = await client.post("/image-analysis/batch", files=[("files", payload)])
        assert first.json()["results"][0]["cached"] is False
        assert second.json()["results"][0]["cached"] is True

    async def test_batch_rejects_over_ten_files(self, client):
        files = [("files", (f"p{i}.png", PNG_MAGIC + b"\x00" * 32, "image/png")) for i in range(11)]
        resp = await client.post("/image-analysis/batch", files=files)
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Analyze-extracted endpoint
# ---------------------------------------------------------------------------


class TestAnalyzeExtractedEndpoint:
    async def test_analyze_extracted_text(self, client, stub_provider):
        resp = await client.post(
            "/image-analysis/analyze-extracted",
            json={"text": "Refer to 2:255 and Sahih al-Bukhari.", "skip_translation": True},
        )
        assert resp.status_code == 200
        analysis = resp.json()["analysis"]
        assert analysis["verses"][0]["reference"] == "2:255"
        assert analysis["hadith"][0]["collection"] == "Sahih al-Bukhari"
        assert analysis["content_type"] == "quran"
