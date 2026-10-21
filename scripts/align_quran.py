"""Stream a JSONL corpus of AlignmentRequest records into aligned results.

Usage: python -m scripts.align_quran requests.jsonl results.jsonl
Each output line has either a result or an error; a bad item does not discard
other records. Any failures produce exit status 1 for automation.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import TextIO

from pydantic import ValidationError

from phoneme_alignment import AlignmentError, AlignmentRequest, align


def align_corpus(records: Iterable[str], output: TextIO) -> int:
    failures = 0
    for index, line in enumerate(records, start=1):
        try:
            request = AlignmentRequest.model_validate_json(line)
            result = align(request.reference, request.acoustics)
            record = {"line": index, "result": result.model_dump(mode="json")}
        except (ValidationError, AlignmentError) as error:
            failures += 1
            record = {"line": index, "error": str(error)}
        output.write(json.dumps(record, ensure_ascii=False) + "\n")
        output.flush()
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("Input and output must be different files")
    with args.input.open(encoding="utf-8") as source, args.output.open("w", encoding="utf-8") as destination:
        return int(align_corpus(source, destination) > 0)


if __name__ == "__main__":
    raise SystemExit(main())
