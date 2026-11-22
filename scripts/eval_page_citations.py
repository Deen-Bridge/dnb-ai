"""Offline evaluation and latency baseline for page-analysis citation extraction (#854).

Two reports, both deterministic and free of network calls or credentials:

  evaluation (default)
      Replays the labeled synthetic pages in
      ``tests/fixtures/page_citation_eval.json`` through
      ``page_analysis.extract_and_verify_references`` and reports:

      precision / recall
          Over accepted references, matched on (type, reference, block index,
          status). A reference with the right location but the wrong status
          counts as both a false positive and a false negative, so promoting an
          unverified match to "validated" is penalized.
      invalid-reference rejection rate
          Of the citations labeled as impossible, how many were rejected.
      provenance accuracy
          Of the accepted references, how many slice back to their matched text
          via block.text[start:end].
      false positives on no-reference pages
          Accepted references on pages labeled as containing none.

  --bench
      Times ``analyze_page`` on fixed synthetic pages at documented sizes and
      prints p50/p95 latency. The analyzer makes no provider calls, so provider
      cost on this path is zero by construction; this says nothing about the
      cost or latency of a real OCR engine, which must be measured separately
      before one is enabled.

Usage:
    python scripts/eval_page_citations.py
    python scripts/eval_page_citations.py --bench --runs 50
    python scripts/eval_page_citations.py --json

The fixture is small and synthetic. These numbers show the contract holds on
known shapes; they are not an estimate of accuracy on real manuscripts.
Exit code is 1 when an evaluation threshold is missed.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from page_analysis import PageBlock, PageInput, analyze_page, extract_and_verify_references  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "page_citation_eval.json"

# Gates. The fixture is curated, so anything below these is a regression.
THRESHOLDS = {
    "precision": 1.0,
    "recall": 1.0,
    "invalid_rejection_rate": 1.0,
    "provenance_accuracy": 1.0,
}

# (blocks, characters per block) for the latency baseline.
BENCH_SHAPES: tuple[tuple[int, int], ...] = ((40, 120), (400, 120), (2000, 100))

_BENCH_LINE = "Be patient, as in Surah 2:153 and [البقرة: ١٥٣]; see Bukhari #1. "


def load_cases(path: Path = FIXTURE) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return list(json.load(handle)["cases"])


def _blocks(case: dict[str, Any]) -> list[PageBlock]:
    return [
        PageBlock(x=0, y=index * 10, width=100, height=10, text=block["text"], confidence=block.get("confidence"))
        for index, block in enumerate(case["blocks"])
    ]


def evaluate(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Score the analyzer against labeled cases. Pure function: no I/O."""
    true_positive = false_positive = false_negative = 0
    invalid_expected = invalid_rejected = 0
    provenance_ok = provenance_total = 0
    no_reference_false_positives = 0
    review_flags = 0
    by_status: dict[str, int] = {"validated": 0, "unverified": 0}
    failures: list[dict[str, Any]] = []

    for case in cases:
        blocks = _blocks(case)
        accepted, rejected = extract_and_verify_references(blocks)

        got = [(ref.type, ref.reference, ref.block_index, ref.status.value) for ref in accepted]
        expected = [tuple(item) for item in case["expected"]]
        remaining = list(expected)
        for item in got:
            if item in remaining:
                remaining.remove(item)
                true_positive += 1
            else:
                false_positive += 1
        false_negative += len(remaining)

        for ref in accepted:
            provenance_total += 1
            provenance_ok += blocks[ref.block_index].text[ref.start : ref.end] == ref.text
            by_status[ref.status.value] += 1
            review_flags += ref.requires_review

        rejected_texts = [item.text for item in rejected]
        for text in case["expected_rejected"]:
            invalid_expected += 1
            invalid_rejected += text in rejected_texts

        if case["category"] == "no_reference":
            no_reference_false_positives += len(accepted)

        if sorted(got) != sorted(expected) or sorted(rejected_texts) != sorted(case["expected_rejected"]):
            failures.append({"id": case["id"], "got": got, "expected": expected, "rejected": rejected_texts})

    def ratio(numerator: int, denominator: int) -> float:
        return round(numerator / denominator, 4) if denominator else 1.0

    return {
        "cases": len(cases),
        "categories": sorted({case["category"] for case in cases}),
        "true_positives": true_positive,
        "false_positives": false_positive,
        "false_negatives": false_negative,
        "precision": ratio(true_positive, true_positive + false_positive),
        "recall": ratio(true_positive, true_positive + false_negative),
        "invalid_expected": invalid_expected,
        "invalid_rejection_rate": ratio(invalid_rejected, invalid_expected),
        "provenance_accuracy": ratio(provenance_ok, provenance_total),
        "no_reference_false_positives": no_reference_false_positives,
        "accepted_by_status": by_status,
        "requires_review": review_flags,
        "failures": failures,
    }


def _bench_page(blocks: int, chars: int) -> PageInput:
    line = (_BENCH_LINE * (chars // len(_BENCH_LINE) + 1))[:chars]
    per_column = max(1, blocks // 2)
    return PageInput(
        width=1000,
        height=20000,
        rtl=True,
        blocks=[
            PageBlock(
                x=50 if index < per_column else 550,
                y=(index % per_column) * 9,
                width=400,
                height=8,
                text=line,
                font_size=12,
                confidence=0.9,
            )
            for index in range(blocks)
        ],
    )


def benchmark(runs: int = 30) -> dict[str, Any]:
    """p50/p95 wall-clock latency of analyze_page on fixed synthetic pages."""
    analyze_page(_bench_page(4, 120))  # warm the surah and hadith indexes once
    results = []
    for blocks, chars in BENCH_SHAPES:
        page = _bench_page(blocks, chars)
        timings = []
        for _ in range(runs):
            started = time.perf_counter()
            analyze_page(page)
            timings.append((time.perf_counter() - started) * 1000)
        timings.sort()
        results.append(
            {
                "blocks": blocks,
                "chars_per_block": chars,
                "total_chars": blocks * chars,
                "runs": runs,
                "p50_ms": round(statistics.median(timings), 2),
                "p95_ms": round(timings[max(0, math.ceil(0.95 * runs) - 1)], 2),
            }
        )
    return {
        "environment": {
            "python": platform.python_version(),
            "machine": platform.machine(),
            "system": platform.system(),
        },
        "provider_calls": 0,
        "provider_cost_usd": 0.0,
        "shapes": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bench", action="store_true", help="also run the latency baseline")
    parser.add_argument("--runs", type=int, default=30, help="timed runs per benchmark shape")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    args = parser.parse_args()

    report: dict[str, Any] = {"evaluation": evaluate(load_cases())}
    if args.bench:
        report["benchmark"] = benchmark(args.runs)

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        ev = report["evaluation"]
        print(f"cases={ev['cases']} categories={', '.join(ev['categories'])}")
        print(
            f"precision={ev['precision']} recall={ev['recall']} (tp={ev['true_positives']}, "
            f"fp={ev['false_positives']}, fn={ev['false_negatives']})"
        )
        print(f"invalid_rejection_rate={ev['invalid_rejection_rate']} of {ev['invalid_expected']}")
        print(f"provenance_accuracy={ev['provenance_accuracy']}")
        print(f"no_reference_false_positives={ev['no_reference_false_positives']}")
        print(f"accepted_by_status={ev['accepted_by_status']} requires_review={ev['requires_review']}")
        for failure in ev["failures"]:
            print(f"FAIL {failure}")
        if "benchmark" in report:
            bench = report["benchmark"]
            print(f"benchmark env={bench['environment']} provider_calls={bench['provider_calls']}")
            for shape in bench["shapes"]:
                print(
                    f"  blocks={shape['blocks']} chars={shape['total_chars']} runs={shape['runs']} "
                    f"p50={shape['p50_ms']}ms p95={shape['p95_ms']}ms"
                )

    missed = [name for name, floor in THRESHOLDS.items() if report["evaluation"][name] < floor]
    if missed or report["evaluation"]["no_reference_false_positives"]:
        print(f"threshold missed: {missed or ['no_reference_false_positives']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
