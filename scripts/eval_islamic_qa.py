"""Islamic QA Benchmark Evaluation Harness.

Evaluates AI models and QA pipelines against the curated Islamic QA Benchmark Dataset (530 items across 10 domains).

Supports:
1. Dataset validation & Citation integrity checking (Quran & Hadith index verification).
2. Offline model scoring (keyword rubrics, key points coverage, abstention checks).
3. Live API evaluation against `/chat` endpoint.
4. Detailed domain scorecards and report generation.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_DATASET = ROOT / "data" / "eval" / "islamic_qa_benchmark.jsonl"
SURAH_INDEX_FILE = ROOT / "data" / "quran" / "surah_index.json"

REQUIRED_DOMAINS = [
    "aqeedah",
    "fiqh_ibadat",
    "fiqh_muamalat",
    "fiqh_munakahat_mirath",
    "ulum_al_quran",
    "mustalah_al_hadith",
    "seerah",
    "tarikh_islami",
    "tasawwuf_adab_akhlaq",
    "contemporary_issues",
]

CANONICAL_HADITH_COLLECTIONS = {
    "bukhari",
    "muslim",
    "abudawud",
    "tirmidhi",
    "nasai",
    "ibnmajah",
    "ahmad",
    "malik",
    "darimi",
}


class SurahIndex:
    def __init__(self, path: Path = SURAH_INDEX_FILE) -> None:
        self.surahs: dict[int, int] = {}
        if path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data:
                        s_num = item.get("id") or item.get("number")
                        ayah_count = item.get("total_verses") or item.get("ayah_count") or item.get("verses_count")
                        if s_num and ayah_count:
                            self.surahs[int(s_num)] = int(ayah_count)
            except Exception as e:
                print(f"[Warning] Failed to load surah_index.json: {e}", file=sys.stderr)

    def is_valid_ayah(self, surah: int, ayah: int) -> bool:
        if not (1 <= surah <= 114):
            return False
        if surah in self.surahs:
            return 1 <= ayah <= self.surahs[surah]
        return True


def load_dataset(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Benchmark dataset not found at {path}")
    items: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num} of {path}: {e}") from e
    return items


def validate_dataset_integrity(records: list[dict[str, Any]]) -> dict[str, Any]:
    surah_idx = SurahIndex()
    errors: list[str] = []
    warnings: list[str] = []

    domain_counts: Counter[str] = Counter()
    difficulty_counts: Counter[str] = Counter()
    citation_types: Counter[str] = Counter()
    total_citations = 0

    seen_ids = set()

    for idx, r in enumerate(records):
        item_id = r.get("id", f"item_{idx}")
        if item_id in seen_ids:
            errors.append(f"Duplicate ID found: {item_id}")
        seen_ids.add(item_id)

        # Domain check
        domain = r.get("domain")
        if not domain or domain not in REQUIRED_DOMAINS:
            errors.append(f"[{item_id}] Invalid or missing domain: {domain}")
        else:
            domain_counts[domain] += 1

        # Difficulty check
        difficulty = r.get("difficulty")
        if difficulty not in ("easy", "medium", "hard"):
            errors.append(f"[{item_id}] Invalid difficulty: {difficulty}")
        else:
            difficulty_counts[difficulty] += 1

        # Question and Expected answer
        if not r.get("question"):
            errors.append(f"[{item_id}] Missing question")
        if not r.get("expected_answer"):
            errors.append(f"[{item_id}] Missing expected_answer")

        # Evaluation criteria check
        eval_crit = r.get("evaluation_criteria", {})
        if not isinstance(eval_crit, dict):
            errors.append(f"[{item_id}] evaluation_criteria must be a dictionary")

        # Citation validation
        citations = r.get("citations", [])
        total_citations += len(citations)
        for c in citations:
            c_type = c.get("type")
            citation_types[c_type or "unknown"] += 1
            if c_type == "quran":
                surah = c.get("surah")
                ayah_start = c.get("ayah_start")
                if surah is None or not (1 <= surah <= 114):
                    errors.append(f"[{item_id}] Invalid Quran surah number: {surah}")
                elif ayah_start is not None and not surah_idx.is_valid_ayah(surah, ayah_start):
                    errors.append(f"[{item_id}] Invalid Quran ayah {ayah_start} for Surah {surah}")
            elif c_type == "hadith":
                collection = c.get("collection", "").lower()
                if collection not in CANONICAL_HADITH_COLLECTIONS:
                    warnings.append(f"[{item_id}] Non-canonical hadith collection: {collection}")

    missing_domains = [d for d in REQUIRED_DOMAINS if domain_counts[d] == 0]
    if missing_domains:
        errors.append(f"Missing required domains: {missing_domains}")

    return {
        "valid": len(errors) == 0,
        "total_records": len(records),
        "errors": errors,
        "warnings": warnings,
        "domain_counts": dict(domain_counts),
        "difficulty_counts": dict(difficulty_counts),
        "citation_types": dict(citation_types),
        "total_citations": total_citations,
    }


def evaluate_single_response(record: dict[str, Any], candidate_answer: str) -> dict[str, Any]:
    """Score candidate answer against expected key points, must-includes, and abstention rules."""
    text_lower = candidate_answer.lower()

    # 1. Abstention check
    requires_abstention = record.get("requires_abstention", False)
    abstention_indicators = [
        "abstain",
        "consult",
        "arbitrat",
        "shariah court",
        "judge",
        "qadi",
        "mufti",
        "dar al-ifta",
        "local scholar",
        "cannot issue",
        "refuse",
        "الامتناع",
        "قاض",
        "محكمة",
        "دار الإفتاء",
    ]

    did_abstain = any(ind in text_lower for ind in abstention_indicators)
    if requires_abstention:
        abstention_score = 1.0 if did_abstain else 0.0
    else:
        abstention_score = 1.0  # not required

    # 2. Must-include keywords
    eval_crit = record.get("evaluation_criteria", {})
    must_include = eval_crit.get("must_include", [])
    included_count = 0
    for term in must_include:
        # Case-insensitive substring or regex search
        pattern = re.escape(term.lower())
        if re.search(pattern, text_lower):
            included_count += 1
    must_include_precision = (included_count / len(must_include)) if must_include else 1.0

    # 3. Must-not-include (prohibitions)
    must_not_include = eval_crit.get("must_not_include", [])
    violations = 0
    for term in must_not_include:
        if re.search(re.escape(term.lower()), text_lower):
            violations += 1
    must_not_include_pass = violations == 0

    # 4. Key points recall
    key_points = record.get("key_points", [])
    matched_kps = 0
    for kp in key_points:
        words = [w.lower() for w in re.findall(r"\w+", kp) if len(w) > 3]
        if not words:
            matched_kps += 1
            continue
        # If at least 40% of salient words in key point appear in answer
        hits = sum(1 for w in words if w in text_lower)
        if hits / len(words) >= 0.4:
            matched_kps += 1
    kp_recall = (matched_kps / len(key_points)) if key_points else 1.0

    # Overall composite score (0.0 to 1.0)
    if requires_abstention:
        composite_score = abstention_score
    else:
        penalty = 0.5 if not must_not_include_pass else 1.0
        composite_score = (0.5 * must_include_precision + 0.5 * kp_recall) * penalty

    is_pass = composite_score >= 0.70

    return {
        "id": record["id"],
        "domain": record["domain"],
        "difficulty": record["difficulty"],
        "composite_score": round(composite_score, 3),
        "passed": is_pass,
        "must_include_precision": round(must_include_precision, 3),
        "must_not_include_pass": must_not_include_pass,
        "key_point_recall": round(kp_recall, 3),
        "abstention_correct": abstention_score == 1.0,
    }


def query_live_api(base_url: str, question: str) -> str:
    """Send query to the live API endpoint."""
    url = f"{base_url.rstrip('/')}/chat"
    payload = json.dumps({"message": question}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("response") or data.get("message") or str(data)
    except Exception as e:
        return f"[API Error: {e}]"


def run_benchmark_eval(
    records: list[dict[str, Any]],
    live_url: str | None = None,
    domain_filter: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    if domain_filter:
        records = [r for r in records if r.get("domain") == domain_filter]
    if limit:
        records = records[:limit]

    results: list[dict[str, Any]] = []
    domain_scores: dict[str, list[float]] = defaultdict(list)
    domain_passes: dict[str, list[bool]] = defaultdict(list)

    for r in records:
        if live_url:
            answer = query_live_api(live_url, r["question"])
        else:
            # Offline mock: evaluate against ground truth expected answer for baseline calibration
            answer = r["expected_answer"]

        res = evaluate_single_response(r, answer)
        results.append(res)
        domain_scores[r["domain"]].append(res["composite_score"])
        domain_passes[r["domain"]].append(res["passed"])

    total_evaluated = len(results)
    overall_avg_score = (sum(r["composite_score"] for r in results) / total_evaluated) if total_evaluated else 0.0
    overall_pass_rate = (sum(1 for r in results if r["passed"]) / total_evaluated) if total_evaluated else 0.0

    domain_scorecard = {}
    for dom in REQUIRED_DOMAINS:
        if dom in domain_scores and domain_scores[dom]:
            scores = domain_scores[dom]
            passes = domain_passes[dom]
            domain_scorecard[dom] = {
                "count": len(scores),
                "average_score": round(sum(scores) / len(scores), 3),
                "pass_rate": round(sum(1 for p in passes if p) / len(passes), 3),
            }

    return {
        "total_evaluated": total_evaluated,
        "overall_average_score": round(overall_avg_score, 3),
        "overall_pass_rate": round(overall_pass_rate, 3),
        "domain_scorecard": domain_scorecard,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Islamic QA Benchmark Dataset")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Path to benchmark jsonl dataset")
    parser.add_argument("--validate-only", action="store_true", help="Only run schema and citation validation")
    parser.add_argument("--domain", type=str, default=None, help="Filter evaluation to a single domain")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of items to evaluate")
    parser.add_argument("--live", action="store_true", help="Run against a live API endpoint")
    parser.add_argument("--url", type=str, default="http://localhost:8000", help="Live API server base URL")
    parser.add_argument("--output", type=Path, default=None, help="Path to save evaluation output report")

    args = parser.parse_args()

    print(f"Loading benchmark dataset from: {args.dataset}")
    records = load_dataset(args.dataset)
    print(f"Loaded {len(records)} records.")

    print("\n--- Running Dataset Integrity & Citation Validation ---")
    val_report = validate_dataset_integrity(records)

    if not val_report["valid"]:
        print(f"[FAILED] Dataset validation failed with {len(val_report['errors'])} errors:", file=sys.stderr)
        for err in val_report["errors"][:15]:
            print(f"  - {err}", file=sys.stderr)
        if len(val_report["errors"]) > 15:
            print(f"  ... and {len(val_report['errors']) - 15} more errors.", file=sys.stderr)
        sys.exit(1)

    print(f"[PASSED] Dataset Integrity Verified across {len(val_report['domain_counts'])} domains!")
    print(f"Total Citations Verified: {val_report['total_citations']} ({val_report['citation_types']})")
    print(f"Domain Distribution: {val_report['domain_counts']}")

    if args.validate_only:
        print("\nValidation-only mode completed successfully.")
        return

    print("\n--- Running Benchmark Evaluation Harness ---")
    live_target = args.url if args.live else None
    mode_str = f"Live API ({live_target})" if args.live else "Offline Baseline"
    print(f"Mode: {mode_str}")

    report = run_benchmark_eval(
        records=records,
        live_url=live_target,
        domain_filter=args.domain,
        limit=args.limit,
    )

    print(f"\nEvaluated {report['total_evaluated']} questions.")
    print(f"Overall Average Score: {report['overall_average_score'] * 100:.1f}%")
    print(f"Overall Pass Rate:     {report['overall_pass_rate'] * 100:.1f}%")
    print("\n--- Domain Breakdown ---")
    for dom, stats in report["domain_scorecard"].items():
        print(
            f"  - {dom:<25}: Score {stats['average_score'] * 100:.1f}% | Pass Rate {stats['pass_rate'] * 100:.1f}% (N={stats['count']})"
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nSaved evaluation results to: {args.output}")


if __name__ == "__main__":
    main()
