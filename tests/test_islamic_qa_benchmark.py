"""Tests for the Islamic QA Benchmark Dataset and Evaluation Harness."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.eval_islamic_qa import (
    CANONICAL_HADITH_COLLECTIONS,
    DEFAULT_DATASET,
    REQUIRED_DOMAINS,
    SurahIndex,
    evaluate_single_response,
    load_dataset,
    validate_dataset_integrity,
)

ROOT = Path(__file__).resolve().parents[1]
METADATA_FILE = ROOT / "data" / "eval" / "islamic_qa_benchmark_metadata.json"


@pytest.fixture(scope="module")
def benchmark_data() -> list[dict]:
    return load_dataset(DEFAULT_DATASET)


@pytest.fixture(scope="module")
def metadata_data() -> dict:
    with open(METADATA_FILE, encoding="utf-8") as f:
        return json.load(f)


def test_dataset_files_exist():
    assert DEFAULT_DATASET.exists(), f"Missing dataset file: {DEFAULT_DATASET}"
    assert METADATA_FILE.exists(), f"Missing metadata file: {METADATA_FILE}"


def test_minimum_records_threshold(benchmark_data: list[dict]):
    assert len(benchmark_data) >= 500, f"Expected >= 500 items, got {len(benchmark_data)}"
    assert len(benchmark_data) == 530, f"Expected exactly 530 items, got {len(benchmark_data)}"


def test_all_ten_domains_represented(benchmark_data: list[dict]):
    domains = {r["domain"] for r in benchmark_data}
    for expected_domain in REQUIRED_DOMAINS:
        assert expected_domain in domains, f"Missing domain: {expected_domain}"

    # Verify balance across domains
    domain_counts: dict[str, int] = {}
    for r in benchmark_data:
        domain_counts[r["domain"]] = domain_counts.get(r["domain"], 0) + 1

    for domain, count in domain_counts.items():
        assert count >= 45, f"Domain {domain} has only {count} items, expected >= 45"


def test_schema_conformance(benchmark_data: list[dict]):
    required_keys = [
        "id",
        "schema_version",
        "domain",
        "sub_domain",
        "difficulty",
        "language",
        "question",
        "question_ar",
        "question_type",
        "expected_answer",
        "expected_answer_ar",
        "key_points",
        "citations",
        "has_ikhtilaf",
        "ikhtilaf_details",
        "requires_abstention",
        "evaluation_criteria",
        "metadata",
    ]

    for record in benchmark_data:
        for key in required_keys:
            assert key in record, f"Record {record.get('id')} missing key: {key}"
        assert record["schema_version"] == "1.0.0"
        assert record["difficulty"] in ("easy", "medium", "hard")
        assert isinstance(record["key_points"], list)
        assert len(record["key_points"]) > 0
        assert isinstance(record["citations"], list)
        assert isinstance(record["evaluation_criteria"], dict)


def test_bilingual_coverage(benchmark_data: list[dict]):
    for record in benchmark_data:
        # English questions & answers
        assert len(record["question"].strip()) > 5, f"Record {record['id']} has empty English question"
        assert len(record["expected_answer"].strip()) > 5, f"Record {record['id']} has empty English expected answer"

        # Arabic questions & answers
        assert len(record["question_ar"].strip()) > 5, f"Record {record['id']} has empty Arabic question"
        assert len(record["expected_answer_ar"].strip()) > 5, f"Record {record['id']} has empty Arabic expected answer"


def test_citation_integrity(benchmark_data: list[dict]):
    surah_idx = SurahIndex()
    total_citations = 0

    for record in benchmark_data:
        for c in record["citations"]:
            total_citations += 1
            c_type = c.get("type")
            assert c_type in ("quran", "hadith", "scholarly"), f"Invalid citation type: {c_type}"
            if c_type == "quran":
                surah = c.get("surah")
                ayah = c.get("ayah_start")
                assert surah is not None, f"Missing surah in {record['id']}"
                assert 1 <= surah <= 114, f"Invalid surah {surah} in {record['id']}"
                if ayah is not None:
                    assert surah_idx.is_valid_ayah(surah, ayah), (
                        f"Invalid ayah {ayah} for surah {surah} in {record['id']}"
                    )
            elif c_type == "hadith":
                coll = c.get("collection", "").lower()
                assert coll in CANONICAL_HADITH_COLLECTIONS, f"Invalid hadith collection {coll} in {record['id']}"

    assert total_citations >= 1000, f"Expected >= 1000 citations, got {total_citations}"


def test_inter_annotator_agreement_score(metadata_data: dict, benchmark_data: list[dict]):
    avg_iaa = metadata_data.get("average_inter_annotator_agreement", 0.0)
    assert avg_iaa > 0.85, f"Average IAA score {avg_iaa} must be > 0.85"

    for record in benchmark_data:
        iaa = record.get("metadata", {}).get("inter_annotator_agreement", 0.0)
        assert iaa >= 0.85, f"Record {record['id']} IAA {iaa} is below threshold 0.85"


def test_abstention_items_present(benchmark_data: list[dict]):
    abstention_items = [r for r in benchmark_data if r["requires_abstention"]]
    assert len(abstention_items) >= 2, "Expected at least 2 abstention test items"
    for item in abstention_items:
        assert item["question_type"] == "abstention_required"


def test_dataset_integrity_validator_function(benchmark_data: list[dict]):
    report = validate_dataset_integrity(benchmark_data)
    assert report["valid"] is True
    assert len(report["errors"]) == 0
    assert report["total_records"] == 530


def test_offline_eval_scoring():
    sample_record = {
        "id": "test-001",
        "domain": "aqeedah",
        "difficulty": "easy",
        "requires_abstention": False,
        "key_points": [
            "Tawhid al-Rububiyyah affirms Allah alone as Creator.",
            "Tawhid al-Uluhiyyah directs all worship exclusively to Allah.",
        ],
        "evaluation_criteria": {
            "must_include": ["Rububiyyah", "Uluhiyyah", "worship"],
            "must_not_include": ["creation can be worshipped"],
        },
    }

    # Good response
    good_answer = "The categories are Tawhid al-Rububiyyah (Allah is the Creator) and Tawhid al-Uluhiyyah (all worship is dedicated to Allah alone)."
    good_res = evaluate_single_response(sample_record, good_answer)
    assert good_res["passed"] is True
    assert good_res["composite_score"] >= 0.70

    # Bad response with prohibited misconception
    bad_answer = "In this religion, creation can be worshipped alongside Rububiyyah."
    bad_res = evaluate_single_response(sample_record, bad_answer)
    assert bad_res["must_not_include_pass"] is False
    assert bad_res["composite_score"] < 0.70


def test_abstention_eval_scoring():
    abstention_record = {
        "id": "test-abs-001",
        "domain": "contemporary_issues",
        "difficulty": "hard",
        "requires_abstention": True,
        "key_points": ["Must abstain from personal dispute"],
        "evaluation_criteria": {
            "must_include": ["abstain", "arbitration", "judge"],
            "must_not_include": [],
        },
    }

    # Answer that abstains
    abstaining_answer = (
        "As an AI, I must abstain from issuing a judicial verdict. You must consult a qualified judge or Shariah court."
    )
    res = evaluate_single_response(abstention_record, abstaining_answer)
    assert res["abstention_correct"] is True
    assert res["passed"] is True

    # Answer that fails to abstain
    failing_answer = "You are definitely right and the other party is guilty."
    res_fail = evaluate_single_response(abstention_record, failing_answer)
    assert res_fail["abstention_correct"] is False
    assert res_fail["passed"] is False


def test_eval_script_subprocess_validation():
    cmd = [sys.executable, str(ROOT / "scripts" / "eval_islamic_qa.py"), "--validate-only"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0
    assert "[PASSED] Dataset Integrity Verified across 10 domains!" in proc.stdout
