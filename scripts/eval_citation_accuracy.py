"""Citation Accuracy Benchmark (#123).

Measures how accurately, completely, and appropriately the AI system cites
Islamic sources when answering questions. Unlike ``eval_citations.py`` (which
tests the parser's extraction/validation), this benchmark scores the *content*
of citations against scholar-validated ground truth sets.

Metrics reported
----------------
  precision
      Fraction of model citations that match a ground-truth citation.
  recall
      Fraction of ground-truth citations that the model produced.
  F1
      Harmonic mean of precision and recall.
  hallucination rate
      Fraction of model citations that are fabricated or misattributed
      (not present in ground truth and not a valid alternative source).
  format correctness
      Fraction of citations whose reference format is valid for its type.
  authority appropriateness
      Fraction of citations whose authority is appropriate for the question
      domain (e.g. fiqh questions should cite jurists, not just hadith).
  completeness (recall on multi-source questions)
      Recall computed only over records that require multiple supporting
      sources.

Success thresholds (from the issue):
  citation accuracy (precision)  > 0.94
  hallucination rate             < 0.03
  format correctness             > 0.97
  authority appropriateness      > 0.90
  completeness (recall)          > 0.85
  zero critical errors           (misattributed Quran verses / major hadith)

Usage:
    python scripts/eval_citation_accuracy.py
    python scripts/eval_citation_accuracy.py --verbose
    python scripts/eval_citation_accuracy.py --validate-only
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from citations import extract_citations  # noqa: E402
from hadith import normalize_collection  # noqa: E402

DEFAULT_DATASET = ROOT / "data" / "eval" / "citation_accuracy_benchmark.jsonl"

# Success thresholds from the issue.
MIN_PRECISION = 0.94
MAX_HALLUCINATION_RATE = 0.03
MIN_FORMAT_CORRECTNESS = 0.97
MIN_AUTHORITY_APPROPRIATENESS = 0.90
MIN_COMPLETENESS_RECALL = 0.85

# Citation types that must be present in the dataset.
REQUIRED_CITATION_TYPES = ("quran", "hadith", "scholarly", "mixed", "multi_source")

# Authority domains and the authorities appropriate to each.
AUTHORITY_DOMAINS = {
    "aqeedah": {"quran", "hadith", "scholarly"},
    "fiqh": {"quran", "hadith", "scholarly"},
    "tafsir": {"quran", "scholarly"},
    "hadith_sciences": {"hadith", "scholarly"},
    "seerah": {"hadith", "scholarly"},
    "contemporary": {"quran", "hadith", "scholarly"},
}

# Canonical scholarly works by author (used for authority appropriateness).
SCHOLARLY_AUTHORS = {
    "ibn taymiyyah": "classical",
    "ibn qayyim": "classical",
    "al-nawawi": "classical",
    "ibn kathir": "classical",
    "al-ghazali": "classical",
    "al-shafi'i": "classical",
    "ibn hanbal": "classical",
    "abu hanifah": "classical",
    "malik ibn anas": "classical",
    "al-qurtubi": "classical",
    "al-tabari": "classical",
    "ibn hazm": "classical",
    "al-suyuti": "classical",
    "ibn rushd": "classical",
    "al-bukhari": "classical",
    "muslim": "classical",
    "al-tirmidhi": "classical",
    "abu dawud": "classical",
    "al-nasai": "classical",
    "ibn majah": "classical",
    "al-albani": "contemporary",
    "ibn baz": "contemporary",
    "ibn uthaymeen": "contemporary",
    "al-fawzan": "contemporary",
    "al-munajjid": "contemporary",
    "yusuf al-qaradawi": "contemporary",
    "sayyid qutb": "contemporary",
    "muhammad asad": "contemporary",
    "hamza yusuf": "contemporary",
    "yasin qadhi": "contemporary",
    "jonathan brown": "contemporary",
    "khaled abou el fadl": "contemporary",
    "ingrid mattson": "contemporary",
    "tariq ramadan": "contemporary",
}

# Format validation patterns per citation type.
_QURAN_FORMAT = re.compile(r"^\d{1,3}:\d{1,3}(?:-\d{1,3})?$")
_HADITH_FORMAT = re.compile(r"^[A-Za-z' -]+ \d+(?:\.\d+)?$")


def load_dataset(path: Path) -> list[dict]:
    """Load the JSONL dataset, skipping blank lines and // comments."""
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


def validate_dataset_integrity(records: list[dict]) -> dict:
    """Validate the benchmark dataset structure and return a report."""
    errors: list[str] = []
    citation_type_counts: dict[str, int] = {}
    multi_source_count = 0

    for record in records:
        rid = record.get("id", "<missing>")
        ctype = record.get("citation_type")
        if ctype not in REQUIRED_CITATION_TYPES:
            errors.append(f"{rid}: invalid citation_type {ctype!r}")
        else:
            citation_type_counts[ctype] = citation_type_counts.get(ctype, 0) + 1

        if not record.get("question"):
            errors.append(f"{rid}: missing question")
        if not isinstance(record.get("ground_truth"), list) or not record["ground_truth"]:
            errors.append(f"{rid}: ground_truth must be a non-empty list")
        if not isinstance(record.get("answer"), str) or not record["answer"].strip():
            errors.append(f"{rid}: missing answer")

        if record.get("requires_multiple_sources"):
            multi_source_count += 1

        # Validate ground truth citations.
        for gt in record.get("ground_truth", []):
            gt_type = gt.get("type")
            if gt_type == "quran":
                surah = gt.get("surah")
                ayah = gt.get("ayah_start")
                if not isinstance(surah, int) or not 1 <= surah <= 114:
                    errors.append(f"{rid}: invalid ground-truth surah {surah}")
                if not isinstance(ayah, int) or ayah < 1:
                    errors.append(f"{rid}: invalid ground-truth ayah {ayah}")
            elif gt_type == "hadith":
                coll = gt.get("collection", "").lower()
                if normalize_collection(coll) is None:
                    errors.append(f"{rid}: invalid ground-truth hadith collection {coll!r}")
            elif gt_type == "scholarly":
                if not gt.get("work"):
                    errors.append(f"{rid}: scholarly ground truth missing work")
            else:
                errors.append(f"{rid}: invalid ground-truth citation type {gt_type!r}")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "total_records": len(records),
        "citation_type_counts": citation_type_counts,
        "multi_source_count": multi_source_count,
    }


def _citation_key(citation) -> str:
    """Return a canonical string key for a citation object."""
    ctype = getattr(citation, "type", None)
    if ctype == "quran":
        return f"quran:{citation.surah}:{citation.ayah_start}"
    if ctype == "hadith":
        return f"hadith:{citation.collection}:{citation.number}"
    if ctype == "scholarly":
        return f"scholarly:{citation.work}"
    return f"unknown:{str(citation)}"


def _ground_truth_key(gt: dict) -> str:
    """Return a canonical string key for a ground-truth citation dict."""
    ctype = gt.get("type")
    if ctype == "quran":
        return f"quran:{gt.get('surah')}:{gt.get('ayah_start')}"
    if ctype == "hadith":
        coll = normalize_collection(gt.get("collection", ""))
        return f"hadith:{coll}:{gt.get('number')}"
    if ctype == "scholarly":
        return f"scholarly:{gt.get('work')}"
    return f"unknown:{str(gt)}"


def _is_critical_error(citation, gt_keys: set[str]) -> bool:
    """A critical error is a misattributed Quran verse or major hadith."""
    ctype = getattr(citation, "type", None)
    if ctype == "quran":
        # Any Quran citation not in ground truth is critical.
        return _citation_key(citation) not in gt_keys
    if ctype == "hadith":
        # Major collections (Bukhari, Muslim) misattribution is critical.
        coll = getattr(citation, "collection", "").lower()
        if "bukhari" in coll or "muslim" in coll:
            return _citation_key(citation) not in gt_keys
    return False


def _validate_format(citation) -> bool:
    """Check that a citation's reference format is valid for its type."""
    ctype = getattr(citation, "type", None)
    if ctype == "quran":
        ref = getattr(citation, "reference", "")
        return bool(_QURAN_FORMAT.match(ref))
    if ctype == "hadith":
        coll = getattr(citation, "collection", "")
        number = getattr(citation, "number", None)
        if not coll or not number:
            return False
        return bool(_HADITH_FORMAT.match(f"{coll} {number}"))
    if ctype == "scholarly":
        work = getattr(citation, "work", "")
        return bool(work and len(work.strip()) > 2)
    return False


def _authority_appropriate(citation, domain: str) -> bool:
    """Check whether a citation's authority is appropriate for the domain."""
    ctype = getattr(citation, "type", None)
    allowed = AUTHORITY_DOMAINS.get(domain, {"quran", "hadith", "scholarly"})
    if ctype == "quran":
        return "quran" in allowed
    if ctype == "hadith":
        return "hadith" in allowed
    if ctype == "scholarly":
        if "scholarly" not in allowed:
            return False
        author = (getattr(citation, "author", "") or "").lower()
        if not author:
            return True  # No author claimed; can't judge, assume appropriate.
        # Contemporary authorities are appropriate for contemporary issues.
        if domain == "contemporary":
            return True
        # For classical domains, contemporary scholars are still acceptable
        # as long as they are recognized authorities.
        return author in SCHOLARLY_AUTHORS
    return False


def evaluate_single_record(record: dict, answer: str | None = None) -> dict:
    """Score one record's answer against its ground-truth citation set."""
    gt_keys = {_ground_truth_key(gt) for gt in record["ground_truth"]}
    gt_by_type: dict[str, list[dict]] = {}
    for gt in record["ground_truth"]:
        gt_by_type.setdefault(gt.get("type"), []).append(gt)

    # Extract citations from the answer.
    if answer is None:
        answer = record.get("answer", "")
    _, extraction = extract_citations(answer)
    citations = extraction.citations

    model_keys = {_citation_key(c) for c in citations}

    # Precision / recall / F1.
    true_positives = len(model_keys & gt_keys)
    precision = true_positives / len(model_keys) if model_keys else 0.0
    recall = true_positives / len(gt_keys) if gt_keys else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    # Hallucination: model citations not in ground truth.
    hallucinated = model_keys - gt_keys
    hallucination_rate = len(hallucinated) / len(model_keys) if model_keys else 0.0

    # Format correctness.
    format_ok = sum(1 for c in citations if _validate_format(c))
    format_correctness = format_ok / len(citations) if citations else 1.0

    # Authority appropriateness.
    domain = record.get("authority_domain", "aqeedah")
    authority_ok = sum(1 for c in citations if _authority_appropriate(c, domain))
    authority_appropriateness = authority_ok / len(citations) if citations else 1.0

    # Critical errors.
    critical_errors = [c for c in citations if _is_critical_error(c, gt_keys)]

    return {
        "id": record["id"],
        "citation_type": record.get("citation_type"),
        "requires_multiple_sources": bool(record.get("requires_multiple_sources")),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "hallucination_rate": round(hallucination_rate, 4),
        "format_correctness": round(format_correctness, 4),
        "authority_appropriateness": round(authority_appropriateness, 4),
        "critical_errors": len(critical_errors),
        "model_citations": len(citations),
        "ground_truth_citations": len(gt_keys),
    }


def evaluate(records: list[dict], verbose: bool = False) -> int:
    """Run the full benchmark and print a report. Returns exit code."""
    results = [evaluate_single_record(r) for r in records]

    total_model = sum(r["model_citations"] for r in results)
    total_gt = sum(r["ground_truth_citations"] for r in results)
    total_tp = sum(round(r["precision"] * r["model_citations"], 4) for r in results)
    total_hallucinated = sum(round(r["hallucination_rate"] * r["model_citations"], 4) for r in results)
    total_format_ok = sum(round(r["format_correctness"] * r["model_citations"], 4) for r in results)
    total_authority_ok = sum(round(r["authority_appropriateness"] * r["model_citations"], 4) for r in results)
    total_critical = sum(r["critical_errors"] for r in results)

    precision = total_tp / total_model if total_model else 0.0
    recall = total_tp / total_gt if total_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    hallucination_rate = total_hallucinated / total_model if total_model else 0.0
    format_correctness = total_format_ok / total_model if total_model else 0.0
    authority_appropriateness = total_authority_ok / total_model if total_model else 0.0

    # Completeness: recall over multi-source records only.
    multi_results = [r for r in results if r["requires_multiple_sources"]]
    multi_tp = sum(round(r["recall"] * r["ground_truth_citations"], 4) for r in multi_results)
    multi_gt = sum(r["ground_truth_citations"] for r in multi_results)
    completeness = multi_tp / multi_gt if multi_gt else 0.0

    print(f"cases                        {len(records)}")
    print(f"model citations              {total_model}")
    print(f"ground-truth citations       {total_gt}")
    print(f"precision                    {precision:.4f}  (floor {MIN_PRECISION:.2f})")
    print(f"recall                       {recall:.4f}")
    print(f"F1                           {f1:.4f}")
    print(f"hallucination rate           {hallucination_rate:.4f}  (ceiling {MAX_HALLUCINATION_RATE:.2f})")
    print(f"format correctness           {format_correctness:.4f}  (floor {MIN_FORMAT_CORRECTNESS:.2f})")
    print(f"authority appropriateness    {authority_appropriateness:.4f}  (floor {MIN_AUTHORITY_APPROPRIATENESS:.2f})")
    print(f"completeness (multi-source)  {completeness:.4f}  (floor {MIN_COMPLETENESS_RECALL:.2f})")
    print(f"critical errors              {total_critical}")

    if verbose:
        print()
        print(f"{'id':<30}{'type':<14}{'prec':<8}{'rec':<8}{'f1':<8}{'hall':<8}{'fmt':<8}{'auth':<8}{'crit'}")
        for r in results:
            print(
                f"{r['id']:<30}{r['citation_type']:<14}"
                f"{r['precision']:<8}{r['recall']:<8}{r['f1']:<8}"
                f"{r['hallucination_rate']:<8}{r['format_correctness']:<8}"
                f"{r['authority_appropriateness']:<8}{r['critical_errors']}"
            )

    failures = []
    if precision < MIN_PRECISION:
        failures.append(f"precision {precision:.4f} below {MIN_PRECISION:.2f}")
    if hallucination_rate > MAX_HALLUCINATION_RATE:
        failures.append(f"hallucination rate {hallucination_rate:.4f} above {MAX_HALLUCINATION_RATE:.2f}")
    if format_correctness < MIN_FORMAT_CORRECTNESS:
        failures.append(f"format correctness {format_correctness:.4f} below {MIN_FORMAT_CORRECTNESS:.2f}")
    if authority_appropriateness < MIN_AUTHORITY_APPROPRIATENESS:
        failures.append(
            f"authority appropriateness {authority_appropriateness:.4f} below {MIN_AUTHORITY_APPROPRIATENESS:.2f}"
        )
    if completeness < MIN_COMPLETENESS_RECALL:
        failures.append(f"completeness {completeness:.4f} below {MIN_COMPLETENESS_RECALL:.2f}")
    if total_critical > 0:
        failures.append(f"{total_critical} critical errors (misattributed Quran/major hadith)")

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
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--validate-only", action="store_true", help="validate dataset integrity only")
    args = parser.parse_args()

    records = load_dataset(args.dataset)
    report = validate_dataset_integrity(records)

    if args.validate_only:
        if report["valid"]:
            print(
                f"[PASSED] Dataset Integrity Verified: {report['total_records']} records, "
                f"{report['multi_source_count']} multi-source"
            )
            return 0
        for error in report["errors"]:
            print(f"ERROR: {error}")
        return 1

    if not report["valid"]:
        print("Dataset integrity check failed:")
        for error in report["errors"][:20]:
            print(f"  ERROR: {error}")
        return 1

    return evaluate(records, args.verbose)


if __name__ == "__main__":
    raise SystemExit(main())
