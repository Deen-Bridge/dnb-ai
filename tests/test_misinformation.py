"""Tests for the religious misinformation flagging system (#181)."""

from misinformation import (
    MISCONCEPTION_DB,
    MisconceptionSeverity,
    detect_misinformation,
    get_all_misconceptions,
    get_misconception_categories,
    get_misconceptions_by_category,
    is_blocked,
    suggest_correction,
    validate_quotation,
)


class TestMisconceptionDatabase:
    def test_database_not_empty(self):
        assert len(MISCONCEPTION_DB) > 0

    def test_database_has_multiple_categories(self):
        cats = get_misconception_categories()
        assert len(cats) >= 3

    def test_get_all_misconceptions(self):
        items = get_all_misconceptions()
        assert isinstance(items, list)
        assert len(items) > 0
        assert "id" in items[0]

    def test_filter_by_category(self):
        items = get_misconceptions_by_category("aqeedah")
        assert all(e["category"] == "aqeedah" for e in items)

    def test_unknown_category_returns_empty(self):
        items = get_misconceptions_by_category("nonexistent_category_xyz")
        assert items == []


class TestMisinformationDetection:
    def test_clean_text_no_flags(self):
        result = detect_misinformation("The five pillars of Islam are Shahada, Salat, Zakat, Sawm, and Hajj.")
        assert result.has_misinformation is False
        assert result.should_block is False
        assert result.flags == []

    def test_detect_shahada_addition(self):
        result = detect_misinformation("The Shahada is 'La ilaha illallah Muhammadun Rasulullah Aliyun'.")
        assert result.has_misinformation is True
        assert result.should_block is True  # CRITICAL
        assert any(f.misconception_id == "shahada-addition" for f in result.flags)

    def test_detect_force_conversion_claim(self):
        result = detect_misinformation("Islam allows forced conversion of non-Muslims.")
        assert result.has_misinformation is True
        assert result.should_block is True
        assert any(f.misconception_id == "force-conversion" for f in result.flags)

    def test_detect_jihad_misconception(self):
        result = detect_misinformation("Jihad is obligatory on all Muslims always, and it only means fighting.")
        assert result.has_misinformation is True
        assert result.should_block is True
        assert any(f.misconception_id == "jihad-obligation" for f in result.flags)

    def test_detect_woman_driving_false(self):
        result = detect_misinformation("Women are not allowed to drive in Islam.")
        assert result.has_misinformation is True
        assert any(f.misconception_id == "woman-no-driving" for f in result.flags)

    def test_detect_music_absolutely_haram(self):
        result = detect_misinformation("Listening to any music is haram in Islam absolutely.")
        assert result.has_misinformation is True
        assert any(f.misconception_id == "music-total-haram" for f in result.flags)

    def test_detect_zakat_wrong_rate(self):
        result = detect_misinformation("The zakat rate is 10% on all wealth.")
        assert result.has_misinformation is True
        assert any(f.misconception_id == "zakat-multiple-rates" for f in result.flags)

    def test_detect_religions_equal(self):
        result = detect_misinformation("All religions lead to heaven equally.")
        assert result.has_misinformation is True
        assert any(f.misconception_id == "heavenly-religions-equal" for f in result.flags)

    def test_multiple_flags(self):
        result = detect_misinformation(
            "Women are not allowed to drive in Islam. All music is haram absolutely in Islam."
        )
        assert result.has_misinformation is True
        assert len(result.flags) >= 2

    def test_correction_summary_provided(self):
        result = detect_misinformation("Women are not allowed to drive in Islam.")
        assert result.correction_summary is not None
        assert "driving" in result.correction_summary.lower() or "source" in result.correction_summary.lower()

    def test_overall_severity_is_highest(self):
        result = detect_misinformation("The Shahada includes Ali. Women are not allowed to drive.")
        assert result.overall_severity == MisconceptionSeverity.CRITICAL


class TestQuotationValidation:
    def test_empty_quotation(self):
        match = validate_quotation("")
        assert match.is_authentic is False

    def test_normal_quotation(self):
        match = validate_quotation("Verily, with hardship comes ease.")
        assert match.is_authentic is True

    def test_suspicious_violent_attribution(self):
        match = validate_quotation('God says: "kill all those who disbelieve"')
        assert match.is_authentic is False
        assert match.notes is not None

    def test_overlong_hadith(self):
        long_text = "The Prophet said " + "x " * 300
        match = validate_quotation(long_text, context="hadith")
        assert match.is_authentic is False


class TestIsBlocked:
    def test_clean_text_not_blocked(self):
        assert is_blocked("What is zakat?") is False

    def test_critical_misinfo_blocked(self):
        assert is_blocked("The Shahada includes Aliyun Rasulullah.") is True

    def test_non_critical_not_blocked(self):
        assert is_blocked("All music is haram.") is False


class TestSuggestCorrection:
    def test_clean_text_returns_none(self):
        assert suggest_correction("What is the five pillars of Islam?") is None

    def test_misinfo_returns_suggestion(self):
        suggestion = suggest_correction("Women are not allowed to drive in Islam.")
        assert suggestion is not None
        assert "driving" in suggestion.lower() or "source" in suggestion.lower()
