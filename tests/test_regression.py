from __future__ import annotations

import json
from pathlib import Path

import pytest

from regression import RegressionRunner


@pytest.fixture
def sample_dataset(tmp_path: Path) -> Path:
    ds_file = tmp_path / "benchmark.jsonl"
    records = [
        {
            "id": "reg-001",
            "domain": "aqeedah",
            "question": "What is Tawhid?",
            "question_ar": "ما التوحيد؟",
            "expected_answer": "Tawhid is the oneness of Allah.",
            "expected_answer_ar": "التوحيد هو توحيد الله",
            "key_points": ["oneness of Allah"],
            "evaluation_criteria": {
                "must_include": ["oneness"],
                "must_not_include": ["polytheism"],
            },
        },
        {
            "id": "reg-002",
            "domain": "fiqh_ibadat",
            "question": "How many daily prayers?",
            "question_ar": "كم عدد الصلوات؟",
            "expected_answer": "There are five daily prayers.",
            "expected_answer_ar": "الصلوات خمس.",
            "key_points": ["five daily prayers"],
            "evaluation_criteria": {
                "must_include": ["five"],
                "must_not_include": ["ten"],
            },
        },
    ]
    with open(ds_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return ds_file


class TestRegressionRunner:
    def test_run_suite_successful(self, sample_dataset: Path, tmp_path: Path):
        baseline_file = tmp_path / "baseline.json"
        runner = RegressionRunner(dataset_path=sample_dataset, baseline_path=baseline_file)

        def dummy_model(q: str) -> str:
            if "Tawhid" in q:
                return "Tawhid is the oneness of Allah."
            return "There are five daily prayers."

        results = runner.run_suite(dummy_model)
        assert results["total_evaluated"] == 2
        assert results["mean_composite_score"] > 0.8
        assert results["pass_rate"] == 1.0
        assert "aqeedah" in results["domain_scores"]

    def test_baseline_comparison_no_regression(self, sample_dataset: Path, tmp_path: Path):
        baseline_file = tmp_path / "baseline.json"
        runner = RegressionRunner(dataset_path=sample_dataset, baseline_path=baseline_file)

        baseline_data = {
            "total_evaluated": 2,
            "mean_composite_score": 0.95,
            "pass_rate": 1.0,
            "domain_scores": {"aqeedah": 0.95, "fiqh_ibadat": 0.95},
        }
        runner.save_baseline(baseline_data)

        current_results = {
            "total_evaluated": 2,
            "mean_composite_score": 0.96,
            "pass_rate": 1.0,
            "domain_scores": {"aqeedah": 0.96, "fiqh_ibadat": 0.96},
        }

        comp = runner.compare_with_baseline(current_results)
        assert comp["has_baseline"] is True
        assert comp["degraded"] is False
        assert "OK" in comp["alert"]

    def test_baseline_comparison_detects_regression(self, sample_dataset: Path, tmp_path: Path):
        baseline_file = tmp_path / "baseline.json"
        runner = RegressionRunner(dataset_path=sample_dataset, baseline_path=baseline_file)

        baseline_data = {
            "total_evaluated": 2,
            "mean_composite_score": 0.95,
            "pass_rate": 1.0,
            "domain_scores": {"aqeedah": 0.95, "fiqh_ibadat": 0.95},
        }
        runner.save_baseline(baseline_data)

        current_results = {
            "total_evaluated": 2,
            "mean_composite_score": 0.80,
            "pass_rate": 0.5,
            "domain_scores": {"aqeedah": 0.80, "fiqh_ibadat": 0.80},
        }

        comp = runner.compare_with_baseline(current_results, tolerance=0.02)
        assert comp["has_baseline"] is True
        assert comp["degraded"] is True
        assert "REGRESSION DETECTED" in comp["alert"]
