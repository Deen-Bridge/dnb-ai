"""Offline calibration, latency and cost evaluation for Quran recitation analysis (#856).

Runs a small labelled set of transcripts through ``analyze_recitation`` and
checks the confidence band, fallback flag and scholar-review reasons against the
expected labels. It also measures per-call latency and confirms that no network
call is made, so the provider cost is zero.

    python -m scripts.eval_recitation_calibration          # table + summary
    python -m scripts.eval_recitation_calibration --json   # machine-readable

The exit code is 0 only when every case matches its label. It needs no secrets,
no network and no ASR.
"""

from __future__ import annotations

import argparse
import json
import socket
import statistics
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # allow `python scripts/eval_recitation_calibration.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quran_recitation import CALIBRATION_VERSION, AnalyzeRequest, analyze_recitation

IKHLAS = "قل هو الله أحد الله الصمد لم يلد ولم يولد ولم يكن له كفوا أحد"
FATIHAH_1_5 = "بسم الله الرحمن الرحيم الحمد لله رب العالمين الرحمن الرحيم مالك يوم الدين إياك نعبد وإياك نستعين"
ASR_HIGH = 0.95


@dataclass(frozen=True)
class EvalCase:
    name: str
    transcript: str
    asr_confidence: float | None
    expected_band: str
    expected_fallback: bool
    expected_review: bool
    expected_reasons: tuple[str, ...]
    riwayah_hint: str | None = None


EVAL_CASES: tuple[EvalCase, ...] = (
    EvalCase("clean_high_asr", IKHLAS, ASR_HIGH, "high", False, False, ()),
    EvalCase("clean_unknown_asr", IKHLAS, None, "medium", False, True, ("asr_confidence_unknown",)),
    EvalCase("clean_medium_asr", IKHLAS, 0.6, "medium", False, True, ("asr_confidence_below_high",)),
    EvalCase("clean_low_asr", IKHLAS, 0.2, "low", True, True, ("asr_confidence_below_high",)),
    EvalCase("one_omission", IKHLAS.replace(" الصمد", ""), ASR_HIGH, "high", False, True, ("text_deviations",)),
    EvalCase(
        "heavy_substitution",
        "قل هو الله أحد كتاب الصمد لم سماء ولم بحر ولم يكن له قلم أحد",
        ASR_HIGH,
        "medium",
        False,
        True,
        ("text_deviations",),
    ),
    EvalCase(
        "heavy_substitution_low_asr",
        "قل هو الله أحد كتاب الصمد لم سماء ولم بحر ولم يكن له قلم أحد",
        0.5,
        "low",
        True,
        True,
        ("asr_confidence_below_high", "text_deviations"),
    ),
    EvalCase("hafs_long_mim", FATIHAH_1_5, ASR_HIGH, "high", False, False, ()),
    EvalCase(
        "non_hafs_short_mim",
        FATIHAH_1_5.replace("مالك", "ملك"),
        ASR_HIGH,
        "high",
        False,
        True,
        ("non_hafs_reading",),
    ),
    EvalCase(
        "riwayah_hint_conflict",
        FATIHAH_1_5,
        ASR_HIGH,
        "high",
        False,
        True,
        ("riwayah_hint_conflict",),
        riwayah_hint="warsh",
    ),
    EvalCase("unrelated_arabic", "ذهب الولد إلى المدرسة صباحا", ASR_HIGH, "low", True, False, ("unidentified",)),
    EvalCase("english_only", "say he is god the one", ASR_HIGH, "low", True, False, ("unidentified",)),
)


@dataclass
class CaseResult:
    name: str
    passed: bool
    band: str
    calibrated_confidence: float
    fallback_recommended: bool
    review_required: bool
    reasons: list[str]
    p50_ms: float
    p95_ms: float
    mismatches: list[str] = field(default_factory=list)


@contextmanager
def _network_forbidden() -> Iterator[None]:
    """Fail loudly if anything tries to open a socket: proves the provider cost is zero."""

    def _blocked(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("network access attempted during recitation evaluation")

    original = socket.socket.connect
    socket.socket.connect = _blocked  # type: ignore[method-assign]
    try:
        yield
    finally:
        socket.socket.connect = original  # type: ignore[method-assign]


def _percentile(samples: list[float], pct: float) -> float:
    ordered = sorted(samples)
    index = min(len(ordered) - 1, max(0, round(pct / 100 * (len(ordered) - 1))))
    return ordered[index]


def evaluate_case(case: EvalCase, repeats: int = 20) -> CaseResult:
    request = AnalyzeRequest(
        transcript=case.transcript, asr_confidence=case.asr_confidence, riwayah_hint=case.riwayah_hint
    )
    timings: list[float] = []
    response = analyze_recitation(request)
    for _ in range(repeats):
        start = time.perf_counter()
        analyze_recitation(request)
        timings.append((time.perf_counter() - start) * 1000)

    mismatches: list[str] = []
    if response.confidence_band != case.expected_band:
        mismatches.append(f"band {response.confidence_band} != {case.expected_band}")
    if response.fallback_recommended != case.expected_fallback:
        mismatches.append(f"fallback {response.fallback_recommended} != {case.expected_fallback}")
    if response.review_required != case.expected_review:
        mismatches.append(f"review {response.review_required} != {case.expected_review}")
    if tuple(response.scholar_review.reasons) != case.expected_reasons:
        mismatches.append(f"reasons {response.scholar_review.reasons} != {list(case.expected_reasons)}")
    if response.scholar_review.required != response.review_required:
        mismatches.append("scholar_review.required disagrees with review_required")

    return CaseResult(
        name=case.name,
        passed=not mismatches,
        band=response.confidence_band,
        calibrated_confidence=response.calibrated_confidence,
        fallback_recommended=response.fallback_recommended,
        review_required=response.review_required,
        reasons=list(response.scholar_review.reasons),
        p50_ms=round(statistics.median(timings), 3),
        p95_ms=round(_percentile(timings, 95), 3),
        mismatches=mismatches,
    )


def run_evaluation(repeats: int = 20) -> dict[str, Any]:
    with _network_forbidden():
        results = [evaluate_case(case, repeats) for case in EVAL_CASES]
    all_p95 = [r.p95_ms for r in results]
    return {
        "calibration_version": CALIBRATION_VERSION,
        "cases": len(results),
        "passed": sum(r.passed for r in results),
        "max_p95_ms": max(all_p95),
        "provider_calls": 0,
        "estimated_cost_usd": 0.0,
        "results": [asdict(r) for r in results],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true", help="print the full report as JSON")
    parser.add_argument("--repeats", type=int, default=20, help="timed calls per case")
    args = parser.parse_args(argv)

    report = run_evaluation(args.repeats)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"{'case':<28} {'band':<7} {'calib':>6} {'fb':<3} {'rev':<3} {'p95ms':>7}  reasons")
        for r in report["results"]:
            flag = "" if r["passed"] else "  <-- " + "; ".join(r["mismatches"])
            print(
                f"{r['name']:<28} {r['band']:<7} {r['calibrated_confidence']:>6.3f} "
                f"{'Y' if r['fallback_recommended'] else 'N':<3} {'Y' if r['review_required'] else 'N':<3} "
                f"{r['p95_ms']:>7.3f}  {','.join(r['reasons']) or '-'}{flag}"
            )
        print(
            f"\n{report['calibration_version']}: {report['passed']}/{report['cases']} cases match labels; "
            f"max p95 latency {report['max_p95_ms']:.3f} ms; provider calls {report['provider_calls']}; "
            f"estimated cost ${report['estimated_cost_usd']:.2f}"
        )
    return 0 if report["passed"] == report["cases"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
