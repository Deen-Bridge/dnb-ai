"""Tests for Arabic OCR module (#214)."""

import base64

import pytest
from fastapi.testclient import TestClient

from arabic_ocr import (
    BoundingBox,
    CalligraphyDetection,
    CalligraphyStyle,
    CharacterResult,
    LineResult,
    OCREngine,
    OCREngineConfig,
    PreprocessingConfig,
    PreprocessingProfile,
    WordResult,
    apply_post_processing,
    calculate_confidence_metrics,
    calculate_word_confidence,
    count_diacritics,
    detect_calligraphy_style,
    is_arabic_diacritic,
    normalize_arabic,
    preprocess_manuscript_image,
    process_manuscript_page,
    strip_diacritics,
)
from main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Diacritic Handling Tests
# ---------------------------------------------------------------------------


class TestDiacriticHandling:
    """Tests for Arabic diacritic detection and manipulation."""

    def test_is_arabic_diacritic_fatha(self):
        """Fatha (short a) should be recognized as diacritic."""
        assert is_arabic_diacritic("\u064e") is True

    def test_is_arabic_diacritic_kasra(self):
        """Kasra (short i) should be recognized as diacritic."""
        assert is_arabic_diacritic("\u0650") is True

    def test_is_arabic_diacritic_damma(self):
        """Damma (short u) should be recognized as diacritic."""
        assert is_arabic_diacritic("\u064f") is True

    def test_is_arabic_diacritic_shadda(self):
        """Shadda (gemination) should be recognized as diacritic."""
        assert is_arabic_diacritic("\u0651") is True

    def test_is_arabic_diacritic_sukun(self):
        """Sukun (no vowel) should be recognized as diacritic."""
        assert is_arabic_diacritic("\u0652") is True

    def test_is_arabic_diacritic_regular_letter(self):
        """Regular Arabic letters should not be diacritics."""
        assert is_arabic_diacritic("\u0628") is False  # Ba
        assert is_arabic_diacritic("\u0645") is False  # Meem

    def test_strip_diacritics_basmala(self):
        """Strip diacritics from fully voweled basmala."""
        text = "بِسْمِ اللَّهِ"
        result = strip_diacritics(text)
        assert result == "بسم الله"
        assert "\u064e" not in result  # No fatha
        assert "\u0652" not in result  # No sukun

    def test_strip_diacritics_empty(self):
        """Empty string should return empty."""
        assert strip_diacritics("") == ""

    def test_strip_diacritics_no_diacritics(self):
        """Text without diacritics should be unchanged."""
        text = "محمد"
        assert strip_diacritics(text) == text

    def test_count_diacritics_basmala(self):
        """Count diacritics in basmala."""
        text = "بِسْمِ"
        # Has kasra, sukun, kasra = 3 diacritics
        assert count_diacritics(text) == 3

    def test_count_diacritics_none(self):
        """Text without diacritics should return 0."""
        assert count_diacritics("محمد") == 0


# ---------------------------------------------------------------------------
# Normalization Tests
# ---------------------------------------------------------------------------


class TestNormalization:
    """Tests for Arabic text normalization."""

    def test_normalize_removes_tatweel(self):
        """Tatweel (kashida) should be removed."""
        text = "اللـــه"
        result = normalize_arabic(text)
        assert "\u0640" not in result

    def test_normalize_alef_forms(self):
        """Various alef forms should normalize to plain alef."""
        assert "ا" in normalize_arabic("أحمد")  # Hamza above
        assert "ا" in normalize_arabic("إسلام")  # Hamza below
        assert "ا" in normalize_arabic("آمين")  # Madda

    def test_normalize_preserves_base_text(self):
        """Basic Arabic text without special forms is preserved."""
        text = "كتاب"
        assert normalize_arabic(text) == text

    def test_normalize_whitespace(self):
        """Multiple spaces should collapse to single space."""
        text = "بسم   الله"
        result = normalize_arabic(text)
        assert "   " not in result
        assert " " in result


# ---------------------------------------------------------------------------
# Calligraphy Style Detection Tests
# ---------------------------------------------------------------------------


class TestCalligraphyDetection:
    """Tests for calligraphy style detection."""

    def test_detect_returns_valid_style(self):
        """Detection should return a valid CalligraphyStyle."""
        # Create minimal test image (1x1 white pixel PNG)
        image_data = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
        )
        result = detect_calligraphy_style(image_data)
        assert isinstance(result, CalligraphyDetection)
        assert result.primary_style in CalligraphyStyle

    def test_detect_with_persian_text(self):
        """Persian letters should boost Nastaliq probability."""
        image_data = b"test"
        text_sample = "پاکستان چیست؟"  # Contains Persian-specific letters
        result = detect_calligraphy_style(image_data, text_sample)
        assert result.features_detected.get("persian_letters") is True

    def test_detect_with_quranic_markers(self):
        """Quranic markers should boost Naskh probability."""
        image_data = b"test"
        text_sample = "الله\u06dd"  # End of ayah marker
        result = detect_calligraphy_style(image_data, text_sample)
        assert result.features_detected.get("quranic_markers") is True

    def test_detect_confidence_in_range(self):
        """Confidence should be between 0 and 1."""
        result = detect_calligraphy_style(b"test")
        assert 0.0 <= result.confidence <= 1.0


# ---------------------------------------------------------------------------
# Preprocessing Tests
# ---------------------------------------------------------------------------


class TestPreprocessing:
    """Tests for image preprocessing."""

    def test_preprocessing_returns_operations(self):
        """Preprocessing should log operations applied."""
        config = PreprocessingConfig()
        _, operations = preprocess_manuscript_image(b"test", config)
        assert len(operations) > 0
        assert any("binarize" in op for op in operations)

    def test_preprocessing_profile_aged(self):
        """Aged profile should apply appropriate settings."""
        config = PreprocessingConfig(profile=PreprocessingProfile.AGED)
        _, operations = preprocess_manuscript_image(b"test", config)
        assert any("denoise" in op for op in operations)
        assert any("contrast" in op for op in operations)

    def test_preprocessing_deskew_enabled(self):
        """Deskew should be in operations when enabled."""
        config = PreprocessingConfig(deskew=True)
        _, operations = preprocess_manuscript_image(b"test", config)
        assert "deskew" in operations

    def test_preprocessing_deskew_disabled(self):
        """Deskew should not be in operations when disabled."""
        config = PreprocessingConfig(deskew=False)
        _, operations = preprocess_manuscript_image(b"test", config)
        assert "deskew" not in operations


# ---------------------------------------------------------------------------
# Confidence Calculation Tests
# ---------------------------------------------------------------------------


class TestConfidenceCalculation:
    """Tests for confidence metric calculation."""

    def test_confidence_empty_lines(self):
        """Empty line list should return zero confidence."""
        metrics = calculate_confidence_metrics([])
        assert metrics.overall == 0.0
        assert metrics.character_level == 0.0

    def test_confidence_perfect_scores(self):
        """Perfect character confidences should give high overall."""
        lines = [
            LineResult(
                text="test",
                confidence=1.0,
                words=[
                    WordResult(
                        text="test",
                        confidence=1.0,
                        characters=[
                            CharacterResult(char="ت", confidence=1.0),
                            CharacterResult(char="س", confidence=1.0),
                        ],
                    )
                ],
            )
        ]
        metrics = calculate_confidence_metrics(lines)
        assert metrics.character_level == 1.0
        assert metrics.word_level == 1.0

    def test_confidence_diacritic_weighted(self):
        """Diacritic confidence should be tracked separately."""
        lines = [
            LineResult(
                text="test",
                confidence=0.9,
                words=[
                    WordResult(
                        text="بِ",
                        confidence=0.9,
                        characters=[
                            CharacterResult(char="ب", confidence=0.95, is_diacritic=False),
                            CharacterResult(char="ِ", confidence=0.7, is_diacritic=True),
                        ],
                    )
                ],
            )
        ]
        metrics = calculate_confidence_metrics(lines)
        assert metrics.diacritic_confidence == 0.7

    def test_word_confidence_calculation(self):
        """Word confidence should weight base chars higher than diacritics."""
        word = WordResult(
            text="بِسْمِ",
            confidence=0.9,
            characters=[
                CharacterResult(char="ب", confidence=0.98, is_diacritic=False),
                CharacterResult(char="ِ", confidence=0.70, is_diacritic=True),
                CharacterResult(char="س", confidence=0.95, is_diacritic=False),
                CharacterResult(char="ْ", confidence=0.65, is_diacritic=True),
                CharacterResult(char="م", confidence=0.97, is_diacritic=False),
            ],
        )
        conf = calculate_word_confidence(word)
        # Base chars (0.98+0.95+0.97)/3 = 0.9667 weighted 80%
        # Diacritics (0.70+0.65)/2 = 0.675 weighted 20%
        assert 0.85 < conf < 0.95


# ---------------------------------------------------------------------------
# Post-Processing Tests
# ---------------------------------------------------------------------------


class TestPostProcessing:
    """Tests for Arabic-specific post-processing."""

    def test_allah_ligature_correction(self):
        """Allah should get shadda on lam if missing."""
        text, _, corrections = apply_post_processing("الله", [])
        assert "اللّه" in text
        assert "allah_ligature_shadda" in corrections

    def test_no_double_correction(self):
        """Already correct text should not be modified."""
        text, _, corrections = apply_post_processing("اللّه", [])
        assert "allah_ligature_shadda" not in corrections


# ---------------------------------------------------------------------------
# API Endpoint Tests
# ---------------------------------------------------------------------------


class TestAPIEndpoints:
    """Tests for FastAPI endpoints."""

    def test_calligraphy_styles_endpoint(self):
        """/calligraphy-styles should return all styles."""
        response = client.get("/arabic-ocr/calligraphy-styles")
        assert response.status_code == 200
        styles = response.json()
        assert len(styles) == len(CalligraphyStyle)
        style_values = {s["style"] for s in styles}
        assert "naskh" in style_values
        assert "thuluth" in style_values

    def test_preprocessing_profiles_endpoint(self):
        """/preprocessing-profiles should return all profiles."""
        response = client.get("/arabic-ocr/preprocessing-profiles")
        assert response.status_code == 200
        profiles = response.json()
        profile_values = {p["profile"] for p in profiles}
        assert "aged" in profile_values
        assert "damaged" in profile_values

    def test_engines_endpoint(self):
        """/engines should return supported OCR engines."""
        response = client.get("/arabic-ocr/engines")
        assert response.status_code == 200
        engines = response.json()
        engine_values = {e["engine"] for e in engines}
        assert "google_vision" in engine_values
        assert "tesseract" in engine_values

    def test_detect_style_endpoint(self):
        """POST /detect-style should detect calligraphy style."""
        # 1x1 white pixel PNG
        image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
        response = client.post(
            "/arabic-ocr/detect-style",
            json={"image_base64": image_b64},
        )
        assert response.status_code == 200
        result = response.json()
        assert "primary_style" in result
        assert "confidence" in result

    def test_preprocess_endpoint(self):
        """POST /preprocess should return preprocessed image."""
        image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
        response = client.post(
            "/arabic-ocr/preprocess",
            json={"image_base64": image_b64},
        )
        assert response.status_code == 200
        result = response.json()
        assert "image_base64" in result
        assert "operations_applied" in result

    def test_process_endpoint(self):
        """POST /process should perform full OCR pipeline."""
        image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
        response = client.post(
            "/arabic-ocr/process",
            json={"image_base64": image_b64},
        )
        assert response.status_code == 200
        result = response.json()
        assert "text" in result
        assert "text_normalized" in result
        assert "confidence" in result
        assert "image_hash" in result

    def test_process_invalid_base64(self):
        """Invalid base64 should return validation error."""
        response = client.post(
            "/arabic-ocr/process",
            json={"image_base64": "not-valid-base64!!!"},
        )
        assert response.status_code == 422  # Validation error


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_pipeline():
    """Test the complete OCR pipeline end-to-end."""
    # Minimal PNG image
    image_data = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )

    result = await process_manuscript_page(
        image_data=image_data,
        preprocessing_config=PreprocessingConfig(profile=PreprocessingProfile.AGED),
        ocr_config=OCREngineConfig(engine=OCREngine.GOOGLE_VISION),
        detect_style=True,
    )

    assert result.text is not None
    assert result.image_hash is not None
    assert len(result.image_hash) == 64  # SHA-256
    assert result.calligraphy is not None
    assert 0.0 <= result.confidence.overall <= 1.0


class TestMultiLineAndDegradedInputCoverage:
    """Expanded coverage for multi-line OCR, degraded regions, and error paths (#431)."""

    def test_multiline_post_processing_and_confidence_aggregation(self):
        lines = [
            LineResult(
                text="بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ",
                words=[
                    WordResult(
                        text="بِسْمِ",
                        confidence=0.92,
                        characters=[
                            CharacterResult(char="ب", confidence=0.95, is_diacritic=False),
                            CharacterResult(char="ِ", confidence=0.82, is_diacritic=True),
                        ],
                        diacritic_count=1,
                    ),
                ],
                confidence=0.92,
            ),
            LineResult(
                text="الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ",
                words=[
                    WordResult(
                        text="الْحَمْدُ",
                        confidence=0.88,
                        characters=[
                            CharacterResult(char="ا", confidence=0.91, is_diacritic=False),
                            CharacterResult(char="ْ", confidence=0.79, is_diacritic=True),
                        ],
                        diacritic_count=1,
                    ),
                ],
                confidence=0.88,
            ),
        ]
        raw_text = "\n".join(line.text for line in lines)
        corrected_text, corrected_lines, corrections = apply_post_processing(raw_text, lines)
        assert len(corrected_lines) == 2
        assert "بِسْمِ" in corrected_text
        assert isinstance(corrections, list)

        clean_q = calculate_confidence_metrics(corrected_lines, damaged_regions=None)
        damaged_q = calculate_confidence_metrics(
            corrected_lines,
            damaged_regions=[
                BoundingBox(x=0, y=0, width=20, height=20),
                BoundingBox(x=30, y=30, width=15, height=15),
            ],
        )
        assert damaged_q.damaged_region_penalty == 0.1
        assert damaged_q.overall < clean_q.overall

    def test_word_confidence_diacritic_only_and_empty_lines_metrics(self):
        diacritic_only = WordResult(
            text="ِ",
            confidence=0.9,
            characters=[CharacterResult(char="ِ", confidence=0.9, is_diacritic=True)],
            diacritic_count=1,
        )
        assert calculate_word_confidence(diacritic_only) == 0.5

        empty_metrics = calculate_confidence_metrics([])
        assert empty_metrics.overall == 0.0
        assert empty_metrics.word_level == 0.0
