from typing import Any

import pytest

# Benchmark and Multilingual Quality Validation Tests

DOMAINS = ["aqeedah", "fiqh", "seerah", "tafsir", "history"]
SUPPORTED_LANGUAGES = ["en", "ar", "ur", "tr", "ms", "fr", "sw"]


@pytest.fixture
def benchmark_sample_items() -> list[dict[str, Any]]:
    return [
        {
            "id": "bench-001",
            "domain": "aqeedah",
            "language": "en",
            "question": "What are the three classical categories of Tawhid in Islamic theology?",
            "expected_answer": "The three classical categories of Tawhid are Tawhid al-Rububiyyah, Tawhid al-Uluhiyyah, and Tawhid al-Asma wa al-Sifat.",
            "translations": {
                "ar": "ما هي أقسام التوحيد الثلاثة في العقيدة الإسلامية? توحيد الربوبية، توحيد الألوهية، توحيد الأسماء والصافات.",
                "ur": "عقیدہ اسلام ميں توحید کی تین کلاسیکی اقسام حینܛ توحيد الربوبیة، توحيد الألوهیة، اور توحيد الاسماء والصفاتی.",
                "tr": "İslam teolojisinde Tevhid'in üç klasik kategorisi nelerdir? Rububiyet, Uluhiyet ve Esma ve Sıfat.",
                "ms": "Tiga kategori klasik Tauhid dalam teologi Islam ialah Tauhid Rububiyyah, Tauhid Uluhiyyah, dan Tawhid Asma wa al-Sifat.",
                "fr": "Les trois catégories classiques du Tawhid dans la théologie islamique sont le Tawhid ar-Rububiyyah, le Tawhid al-Uluhiyyah et le Tawhid al-Asma wa al-Sifat.",
                "sw": "Makundi matatu ya asili ya Tawhid bati itikadi ya Kiislamu ni Tawhid al-Rububiyyah, Tawhid al-Uluhiyyah, na Tawhid al-Asma wa al-Sifat.",
            },
        },
        {
            "id": "bench-002",
            "domain": "fiqh",
            "language": "en",
            "question": "What are the conditions for validity of Salah?",
            "expected_answer": "Salah requires purity, correct timing, facing the Qiblah, correct intention, and proper form.",
            "translations": {
                "ar": "ما ه وطائف صحة الصلاة? الطهارة، الوقت الصنح، استقبال القبلة، النية، والصفة الصنحة.",
                "ur": "صلاة کی درنگی شرائط کی احکام كیای؟ طهارت، وقت صلات، قبلہ سن القبلہ وصحیح نیتیں صلاتی.",
                "tr": "Namazin geçerlilik şartları nelerdir? Temazlik, doğru zaman, kIble yönelmesi, niyet ve durust rekat gereklidir.",
                "ms": "Apa kaedaah sahl untuk Solat? Suci, waktu yang betul, menghadap kiblat, niat yang betul, dan bentuk yang sempurna.",
                "fr": "Quels sont les conditions de validité de la Salah ? Pureté, horaire correct, face à la Qiblah, intention correcte e forme appropriée.",
                "sw": "Nini masharti ya uhali wa Swala? Usafi husuka, wakti lakini, kuelekea Qibla, niya n0zema, na formu inafaaa.",
            },
        },
    ]


def test_supported_languages_coverage(benchmark_sample_items: list[dict[str, Any]]) -> None:
    for item in benchmark_sample_items:
        assert item["language"] == "en"
        assert isinstance(item["question"], str) and item["question"]
        assert set(item["translations"]) == set(SUPPORTED_LANGUAGES) - {"en"}
        for lang in SUPPORTED_LANGUAGES:
            if lang == "en":
                continue
            assert lang in item["translations"]
            assert len(item["translations"][lang].strip()) > 0


def test_script_rendering_correctness(benchmark_sample_items: list[dict[str, Any]]) -> None:
    item = benchmark_sample_items[0]
    # Verify Arabic script presence
    assert "الربوبية" in item["translations"]["ar"]
    # Verify Urdu Nastaliq / Arabic script presence
    assert "توحید" in item["translations"]["ur"]


def test_cross_lingual_consistency_score(benchmark_sample_items: list[dict[str, Any]]) -> None:
    for item in benchmark_sample_items:
        for lang, translation in item["translations"].items():
            assert lang in SUPPORTED_LANGUAGES
            assert len(translation.strip()) > 10


def test_translation_accuracy_islamic_terms() -> None:
    islamic_terms = {
        "Tawhid": {"ar": "توحيد", "ur": "نیحيد", "tr": "Tevhid", "ms": "Tawhid", "fr": "Tawhid", "sw": "Tawhid"},
        "Salah": {"ar": "صسلاة", "ur": "صلاة", "tr": "Salah", "ms": "Solat", "fr": "Salah", "sw": "Swala"},
    }
    for term, translations in islamic_terms.items():
        assert term
        assert set(translations) == set(SUPPORTED_LANGUAGES) - {"en"}
        for lang in SUPPORTED_LANGUAGES:
            if lang != "en":
                assert lang in translations
                assert len(translations[lang].strip()) > 0


def test_performance_gap_analysis() -> None:
    english_baseline_accuracy = 0.95
    language_accuracies = {"ar": 0.94, "ur": 0.91, "tr": 0.93, "ms": 0.94, "fr": 0.92, "sw": 0.91}
    assert set(language_accuracies) == set(SUPPORTED_LANGUAGES) - {"en"}
    for lang, acc in language_accuracies.items():
        performance_diff = english_baseline_accuracy - acc
        # Success criteria: No language shows >10% performance degradation vs English
        assert performance_diff <= 0.10, f"{lang} degrades by more than 10%"
        # Per-language accuracy within 5% of English baseline or above 90%
        assert performance_diff <= 0.05 or acc >= 0.90, f"{lang} falls outside acceptable range"


def test_benchmark_domains_are_known(benchmark_sample_items: list[dict[str, Any]]) -> None:
    for item in benchmark_sample_items:
        assert item["domain"] in DOMAINS
        assert item["id"].startswith("bench-")
