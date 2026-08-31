from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
EVAL_DIR = ROOT / "data" / "eval"
DEFAULT_DATASET = EVAL_DIR / "islamic_qa_benchmark.jsonl"
DEFAULT_BASELINE = EVAL_DIR / "baseline_results.json"


class RegressionRunner:
    def __init__(self, dataset_path: Path | str = DEFAULT_DATASET, baseline_path: Path | str = DEFAULT_BASELINE) -> None:
        self.dataset_path = Path(dataset_path)
        self.baseline_path = Path(baseline_path)

    def load_dataset(self) -> list[dict[str, Any]]:
        if not self.dataset_path.exists():
            return []
        records = []
        with open(self.dataset_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return records

    def load_baseline(self) -> dict[str, Any] | None:
        if not self.baseline_path.exists():
            return None
        with open(self.baseline_path, encoding="utf-8") as f:
            return json.load(f)

    def save_baseline(self, results: dict[str, Any]) -> None:
        self.baseline_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.baseline_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    def evaluate_model_output(self, record: dict[str, Any], answer: str) -> dict[str, Any]:
        from scripts.eval_islamic_qa import evaluate_single_response
        return evaluate_single_response(record, answer)

    def run_suite(self, model_callable: Any, max_items: int | None = None) -> dict[str, Any]:
        records = self.load_dataset()
        if max_items is not None:
            records = records[:max_items]

        scores = []
        passed_count = 0
        domain_scores: dict[str, list[float]] = {}

        results_detail = []
        for r in records:
            question = r["question"]
            try:
                answer = model_callable(question)
            except Exception as e:
                answer = f"ERROR: {e}"

            eval_res = self.evaluate_model_output(r, answer)
            score = eval_res.get("composite_score", 0.0)
            passed = eval_res.get("passed", False)

            scores.append(score)
            if passed:
                passed_count += 1

            domain = r.get("domain", "general")
            domain_scores.setdefault(domain, []).append(score)

            results_detail.append({
                "id": r["id"],
                "domain": domain,
                "score": score,
                "passed": passed,
            })

        mean_score = statistics.mean(scores) if scores else 0.0
        pass_rate = (passed_count / len(records)) if records else 0.0

        domain_summary = {
            d: statistics.mean(vals) if vals else 0.0
            for d, vals in domain_scores.items()
        }

        summary = {
            "total_evaluated": len(records),
            "mean_composite_score": round(mean_score, 4),
            "pass_rate": round(pass_rate, 4),
            "domain_scores": domain_summary,
            "details": results_detail,
        }
        return summary

    def compare_with_baseline(self, current_results: dict[str, Any], tolerance: float = 0.02) -> dict[str, Any]:
        baseline = self.load_baseline()
        if not baseline:
            return {
                "has_baseline": False,
                "degraded": False,
                "message": "No baseline found. Current results established as new baseline.",
            }

        base_score = baseline.get("mean_composite_score", 0.0)
        curr_score = current_results.get("mean_composite_score", 0.0)

        diff = curr_score - base_score
        degraded = diff < (-tolerance)

        domain_diffs = {}
        base_domains = baseline.get("domain_scores", {})
        curr_domains = current_results.get("domain_scores", {})
        for d, curr_val in curr_domains.items():
            base_val = base_domains.get(d, curr_val)
            domain_diffs[d] = round(curr_val - base_val, 4)

        return {
            "has_baseline": True,
            "baseline_score": base_score,
            "current_score": curr_score,
            "difference": round(diff, 4),
            "degraded": degraded,
            "tolerance": tolerance,
            "domain_differences": domain_diffs,
            "alert": "REGRESSION DETECTED: Performance dropped below baseline tolerance." if degraded else "OK",
        }
