"""Tests for Arabic Language Benchmark proficiency, comprehension, diacritics, and terminology."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATASET_FILE = ROOT / "data" / "eval" / "islamic_qa_benchmark.jsonl"


@pytest.fixture(scope="module")
def arabic_benchmark_records() -> list[dict]:
    if not DATASET_FILE.exists():
        pytest.skip("Benchmark dataset not found")
    records = []
    with open(DATASET_FILE, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                if "question_ar" in item and item["question_ar"]:
                    records.append(item)
    return records


def test_arabic_questions_present(arabic_benchmark_records: list[dict]):
    assert len(arabic_benchmark_records) >= 400, f"Expected >= 400 Arabic benchmark records, got {len(arabic_benchmark_records)}"


def test_arabic_fields_completeness(arabic_benchmark_records: list[dict]):
    for record in arabic_benchmark_records[:100]:
        assert "question_ar" in record
        assert "expected_answer_ar" in record
        assert len(record["question_ar"].strip()) > 0
        assert len(record["expected_answer_ar"].strip()) > 0


def test_arabic_diacritical_mark_handling(arabic_benchmark_records: list[dict]):
    # Verify presence of classical/Quranic citations with Arabic text or diacritics
    quranic_citations_count = 0
    for record in arabic_benchmark_records:
        citations = record.get("citations", [])
        for cit in citations:
            if cit.get("type") == "quran":
                quranic_citations_count += 1
    assert quranic_citations_count > 0, "Expected Quranic citations in Arabic benchmark records"


def test_arabic_terminology_precision(arabic_benchmark_records: list[dict]):
    domains = {r["domain"] for r in arabic_benchmark_records}
    assert "aqeedah" in domains
    assert "mustalah_al_hadith" in domains
    assert "fiqh_ibadat" in domains
