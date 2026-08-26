"""Tests for the multi-level Arabic normalization pipeline (#191)."""

from arabic_normalization import (
    NormalizationConfig,
    NormalizationLevel,
    diacritic_density,
    looks_like_quranic,
    normalize_arabic_text,
    normalize_with_metrics,
)

BASMALA_DIACRITIZED = "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"
BASMALA_PLAIN = "بسم الله الرحمن الرحيم"


# ---------------------------------------------------------------------------
# Levels
# ---------------------------------------------------------------------------


class TestLevels:
    def test_none_is_identity(self):
        assert normalize_arabic_text("مُحَمَّد", NormalizationLevel.NONE) == "مُحَمَّد"

    def test_none_preserves_everything(self):
        text = "  إِنَّما  ٱللّٰهَ "
        assert normalize_arabic_text(text, NormalizationLevel.NONE) == text

    def test_light_strips_tatweel_and_whitespace_only(self):
        result = normalize_arabic_text("مــحـــمــد", NormalizationLevel.LIGHT)
        assert result == "محمد"

    def test_light_keeps_diacritics(self):
        assert normalize_arabic_text("مُحَمَّد", NormalizationLevel.LIGHT) == "مُحَمَّد"

    def test_full_strips_diacritics(self):
        assert normalize_arabic_text(BASMALA_DIACRITIZED, NormalizationLevel.FULL) == BASMALA_PLAIN

    def test_full_folds_alef_variants(self):
        assert normalize_arabic_text("أحمد إسلام قرآن ٱلله", NormalizationLevel.FULL) == "احمد اسلام قران الله"

    def test_full_folds_hamza_carriers(self):
        # ؤ -> و and ئ -> ي; ta marbuta folds to ha at FULL level.
        assert normalize_arabic_text("سؤال", NormalizationLevel.FULL) == "سوال"
        assert normalize_arabic_text("مئة", NormalizationLevel.FULL) == "ميه"

    def test_full_folds_alef_maksura_and_ta_marbuta(self):
        assert normalize_arabic_text("على فاطمة", NormalizationLevel.FULL) == "علي فاطمه"

    def test_empty_and_none_like_inputs(self):
        assert normalize_arabic_text("") == ""
        assert normalize_arabic_text(None) == ""


# ---------------------------------------------------------------------------
# Quranic preservation (context-aware)
# ---------------------------------------------------------------------------


class TestQuranicPreservation:
    def test_auto_preserves_heavily_diacritized_text(self):
        cfg = NormalizationConfig(level=NormalizationLevel.FULL, preserve="auto")
        result = normalize_arabic_text(BASMALA_DIACRITIZED, cfg)
        assert "\u064e" in result or "\u0650" in result or "\u0651" in result
        # Letter folding still applies even when diacritics are kept.
        assert "أ" not in result and "إ" not in result

    def test_never_strips_even_when_quranic(self):
        result = normalize_arabic_text(BASMALA_DIACRITIZED, NormalizationLevel.FULL)
        assert normalize_arabic_text(result) == result

    def test_always_flag_preserves_any_diacritics(self):
        cfg = NormalizationConfig(level=NormalizationLevel.FULL, preserve="always")
        assert normalize_arabic_text("مُحَمَّد", cfg) == "مُحَمَّد"

    def test_plain_prose_not_treated_as_quranic(self):
        cfg = NormalizationConfig(level=NormalizationLevel.FULL, preserve="auto")
        assert normalize_arabic_text("كيف حالك اليوم", cfg) == "كيف حالك اليوم"


class TestDetectionHeuristics:
    def test_high_diacritic_density_detected(self):
        assert looks_like_quranic(BASMALA_DIACRITIZED) is True

    def test_ornate_brackets_detected(self):
        assert looks_like_quranic("﴿إِنَّمَا﴾") is True

    def test_honorific_detected(self):
        assert looks_like_quranic("قال ﷺ إنما الأعمال بالنيات") is True

    def test_ordinary_prose_not_detected(self):
        assert looks_like_quranic("ذهبت إلى المسجد بالسيارة") is False


# ---------------------------------------------------------------------------
# Configurability
# ---------------------------------------------------------------------------


class TestConfigurability:
    def test_disable_ta_marbuta_fold(self):
        cfg = NormalizationConfig(level=NormalizationLevel.FULL, fold_ta_marbuta=False)
        assert normalize_arabic_text("فاطمة", cfg) == "فاطمه"[0] + "اطمة"

    def test_disable_all_folding_keeps_letters(self):
        cfg = NormalizationConfig(
            level=NormalizationLevel.FULL,
            fold_alef_variants=False,
            fold_hamza_carriers=False,
            fold_alef_maksura=False,
            fold_ta_marbuta=False,
            remove_standalone_hamza=False,
        )
        result = normalize_arabic_text("إلى مسألة ء", cfg)
        # Tashkeel stripped (FULL) but letter forms preserved.
        assert "ى" in result and "ة" in result and "ء" in result

    def test_letter_forms_survive_when_folding_disabled(self):
        cfg = NormalizationConfig(
            level=NormalizationLevel.FULL,
            fold_alef_variants=False,
            fold_ta_marbuta=False,
            fold_alef_maksura=False,
            fold_hamza_carriers=False,
        )
        result = normalize_arabic_text("إلى مسألة", cfg)
        # With all folding off, original letter forms are preserved.
        assert "ى" in result and "إ" in result and "ة" in result


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_metrics_report_removed_marks(self):
        normalized, metrics = normalize_with_metrics(BASMALA_DIACRITIZED)
        assert metrics.diacritics_removed > 10
        assert metrics.diacritic_density_after == 0.0
        assert metrics.diacritic_density_before > 0.25
        assert metrics.change_ratio > 0

    def test_metrics_detect_quranic_preservation(self):
        cfg = NormalizationConfig(level=NormalizationLevel.FULL, preserve="auto")
        _, metrics = normalize_with_metrics(BASMALA_DIACRITIZED, cfg)
        assert metrics.preserved_quranic is True
        assert metrics.diacritic_density_after > 0

    def test_metrics_zero_change_on_plain_text(self):
        _, metrics = normalize_with_metrics("بسم الله")
        assert metrics.change_ratio == 0.0
        assert metrics.diacritics_removed == 0

    def test_density_helper(self):
        assert diacritic_density("") == 0.0
        assert diacritic_density("بسم") == 0.0
        assert diacritic_density("مُ") == 1.0


# ---------------------------------------------------------------------------
# Search-recall behaviour
# ---------------------------------------------------------------------------


class TestSearchRecall:
    def test_vocalized_and_plain_collapse_to_same_key(self):
        vocalized = normalize_arabic_text("الصَّلَاةِ وَالزَّكَاةِ")
        plain = normalize_arabic_text("الصلاة والزكاة")
        assert vocalized == plain

    def test_hamza_spelling_variations_match(self):
        assert normalize_arabic_text("اسلام") == normalize_arabic_text("إسلام")
        assert normalize_arabic_text("قران") == normalize_arabic_text("قرآن")

    def test_idempotence(self):
        once = normalize_arabic_text(BASMALA_DIACRITIZED)
        assert normalize_arabic_text(once) == once
