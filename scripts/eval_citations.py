"""Report how well structured citations are extracted and validated (#15).

Two modes:

  offline (default)
      Replays the recorded answers in the eval set through ``extract_citations``.
      No API key, no network. This is the mode CI and reviewers can run, and it
      is the one that regresses loudly if the parser or the surah index changes.

  --live
      Sends each question to a running API and evaluates the real answers.
      Requires the server to be up and a Gemini key configured on it. Use this
      to measure whether the model actually complies with the block format,
      which the offline mode cannot tell you.

Usage:
    python scripts/eval_citations.py
    python scripts/eval_citations.py --live --url http://localhost:8000
    python scripts/eval_citations.py --verbose

Three metrics are reported:

  extraction rate
      Of the cases that should produce citations, how many did. Cases that
      correctly cite nothing are excluded, so an abstention is not a failure.

  validity rate
      Of the genuine citations in the set, how many survived validation. The
      deliberately fabricated ones (``expected_rejections``) are excluded from
      the denominator -- otherwise the negative fixtures would drag the rate
      down and the eval would punish the parser for doing its job.

  rejection rate
      Of the deliberately fabricated citations, how many were rejected. This is
      the metric that would catch a parser that validated everything blindly,
      which a validity rate alone cannot see.

Exit code is 1 when a threshold is missed, so this can gate a release.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from citations import extract_citations  # noqa: E402

DEFAULT_DATASET = ROOT / "data" / "eval" / "citations_eval.jsonl"

# An answer that cites nothing is not a parser failure, so extraction rate is
# measured only over cases the eval set says should produce citations.
MIN_EXTRACTION_RATE = 0.90
MIN_VALIDITY_RATE = 0.90

# Rejecting a fabricated citation is deterministic, not statistical: the surah
# index either has 200 surahs or it does not. So the floor here is absolute.
MIN_REJECTION_RATE = 1.00

# LLM-as-judge configuration
JUDGE_DIMENSIONS = ["accuracy", "completeness", "appropriateness", "citation_quality", "tone"]
JUDGE_SCALE = 5
DEFAULT_JUDGE_URL = "http://localhost:8000/judge"


def load_dataset(path):
    if not path.exists():
        sys.exit(f"Dataset not found: {path}")
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            sys.exit(f"{path}:{line_number}: invalid JSON ({exc})")
    return records


def fetch_live_answer(url, question):
    """Ask a running server. Imported lazily so offline mode needs no httpx."""
    import httpx

    response = httpx.post(
        f"{url.rstrip('/')}/chat",
        json={"prompt": question},
        timeout=60.0,
    )
    response.raise_for_status()
    return response.json()


def fetch_judge_scores(judge_url, question, answer):
    """Get LLM judge scores for a single Q&A pair."""
    import httpx
    response = httpx.post(
        judge_url,
        json={
            "question": question,
            "answer": answer,
            "dimensions": JUDGE_DIMENSIONS,
            "scale": JUDGE_SCALE,
        },
        timeout=60.0,
    )
    response.raise_for_status()
    return response.json()


def evaluate_with_judge(records, judge_url, verbose=False):
    """Evaluate records using an LLM judge."""
    scores = {dim: [] for dim in JUDGE_DIMENSIONS}
    rows = []
    for record in records:
        question = record["question"]
        answer = record.get("answer", "")
        if not answer and "response" in record:
            answer = record["response"]
        result = fetch_judge_scores(judge_url, question, answer)
        row = {"id": record["id"]}
        for dim in JUDGE_DIMENSIONS:
            score = result.get(dim)
            row[dim] = score
            if score is not None:
                scores[dim].append(float(score))
        rows.append(row)

    print("LLM-as-Judge Evaluation")
    print(f"mode              judge")
    print(f"cases             {len(records)}")
    for dim in JUDGE_DIMENSIONS:
        vals = scores[dim]
        avg = sum(vals) / len(vals) if vals else 0.0
        print(f"{dim:20s} {avg:.2f} (n={len(vals)})")

    if verbose:
        print()
        header = f"{'id':<30}" + "".join(f"{dim:<20}" for dim in JUDGE_DIMENSIONS)
        print(header)
        for row in rows:
            line = f"{row['id']:<30}" + "".join(f"{str(row[dim]):<20}" for dim in JUDGE_DIMENSIONS)
            print(line)

    return 0


def evaluate(records, live_url=None, verbose=False):
    expected_citing = 0
    produced_citations = 0
    total_attempted = 0
    total_valid = 0
    total_genuine = 0
    total_planted = 0
    total_rejected = 0
    leaks = []
    rows = []

    for record in records:
        question = record["question"]
        expects = bool(record.get("expects_citations", True))

        if live_url:
            body = fetch_live_answer(live_url, question)
            prose = body.get("response") or ""
            citations = body.get("citations") or []
            # The server already parsed; re-parse the prose only to catch leaks.
            _, residue = extract_citations(prose)
            attempted = len(citations)
            valid = len(citations)
            if residue.attempted:
                leaks.append(record["id"])
        else:
            prose, extraction = extract_citations(record["answer"])
            citations = extraction.citations
            attempted = extraction.attempted
            valid = len(extraction.citations)
            if "<<<CITATIONS>>>" in prose or "<<<END_CITATIONS>>>" in prose:
                leaks.append(record["id"])

        # Citations the fixture deliberately fabricated. They are not parser
        # failures, they are the negative half of the test set.
        planted = int(record.get("expected_rejections", 0))

        if expects:
            expected_citing += 1
            if valid:
                produced_citations += 1

        total_attempted += attempted
        total_valid += valid
        total_genuine += max(attempted - planted, 0)
        total_planted += planted
        total_rejected += max(attempted - valid, 0)

        rows.append(
            {
                "id": record["id"],
                "expects": expects,
                "attempted": attempted,
                "valid": valid,
                "planted": planted,
                "note": record.get("note", ""),
            }
        )

    extraction_rate = produced_citations / expected_citing if expected_citing else 1.0
    validity_rate = total_valid / total_genuine if total_genuine else 1.0
    # Capped at the planted count: over-rejection is a validity failure, not a
    # rejection success, and validity_rate is where it should surface.
    rejection_rate = min(total_rejected, total_planted) / total_planted if total_planted else 1.0

    print(f"mode              {'live' if live_url else 'offline'}")
    print(f"cases             {len(records)}")
    print(f"expected to cite  {expected_citing}")
    print(f"did cite          {produced_citations}")
    print(f"citations parsed  {total_attempted}  ({total_planted} planted invalid)")
    print(f"citations valid   {total_valid} of {total_genuine} genuine")
    print(f"fabrications cut  {total_rejected} of {total_planted} planted")
    print(f"extraction rate   {extraction_rate:.0%}  (floor {MIN_EXTRACTION_RATE:.0%})")
    print(f"validity rate     {validity_rate:.0%}  (floor {MIN_VALIDITY_RATE:.0%})")
    print(f"rejection rate    {rejection_rate:.0%}  (floor {MIN_REJECTION_RATE:.0%})")
    print(f"marker leaks      {len(leaks)}")

    if verbose:
        print()
        print(f"{'id':<30}{'cites':<7}{'parsed':<8}{'valid':<7}{'planted':<9}note")
        for row in rows:
            print(
                f"{row['id']:<30}{str(row['expects']):<7}"
                f"{row['attempted']:<8}{row['valid']:<7}{row['planted']:<9}{row['note']}"
            )

    failures = []
    if extraction_rate < MIN_EXTRACTION_RATE:
        failures.append(f"extraction rate {extraction_rate:.0%} below {MIN_EXTRACTION_RATE:.0%}")
    if validity_rate < MIN_VALIDITY_RATE:
        failures.append(f"validity rate {validity_rate:.0%} below {MIN_VALIDITY_RATE:.0%}")
    if rejection_rate < MIN_REJECTION_RATE:
        failures.append(
            f"rejection rate {rejection_rate:.0%} below {MIN_REJECTION_RATE:.0%}: a fabricated citation was accepted"
        )
    if leaks:
        failures.append(f"citation markers leaked into prose for: {', '.join(leaks)}")

    if failures:
        print()
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1

    print("\nPASS")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--live", action="store_true", help="query a running server")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--judge", action="store_true", help="run LLM-as-judge evaluation")
    parser.add_argument("--judge-url", default=DEFAULT_JUDGE_URL, help="LLM judge endpoint URL")
    args = parser.parse_args()

    records = load_dataset(args.dataset)
    if args.judge:
        return evaluate_with_judge(records, args.judge_url, args.verbose)
    return evaluate(records, args.url if args.live else None, args.verbose)


if __name__ == "__main__":
    raise SystemExit(main())
