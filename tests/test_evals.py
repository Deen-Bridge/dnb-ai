import json
from pathlib import Path

from evals.run import evaluate_deterministic, load_dataset

ROOT = Path(__file__).resolve().parents[1]


def test_dataset_covers_required_categories_and_has_at_least_thirty_cases():
    records = load_dataset(ROOT / "evals" / "dataset.jsonl")
    categories = {record["category"] for record in records}

    assert len(records) >= 30
    assert {
        "aqeedah_basics",
        "fiqh_prayer",
        "fiqh_fasting",
        "fiqh_zakat",
        "seerah_facts",
        "quran_sources",
        "deflection_scope",
        "adversarial_refusal",
    } <= categories


def test_deterministic_checks_accept_a_structured_source_and_refusal():
    body = {
        "response": (
            "I cannot help issue a binding fatwa. I can provide general educational information. "
            "Ayat al-Kursi is in Surah Al-Baqarah 2:255."
        ),
        "moderation": {"action": "refuse"},
        "citations": [{"type": "quran", "surah": 2, "ayah_start": 255, "ayah_end": None}],
    }
    expectations = {
        "required_refusal": True,
        "required_scholar_referral": False,
        "required_patterns": ["educational"],
        "required_surah_refs": [{"surah": 2, "ayah": 255}],
        "require_structured_citations": True,
    }

    result = evaluate_deterministic(body, expectations)

    assert result["passed"] is True
    assert result["checks"]["required_surah_refs"]["structured_matches"] == [True]


def test_dataset_is_valid_jsonl():
    for line in (ROOT / "evals" / "dataset.jsonl").read_text(encoding="utf-8").splitlines():
        assert isinstance(json.loads(line), dict)
