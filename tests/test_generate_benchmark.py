"""Tests for the hallucination benchmark data generator."""

import json
from pathlib import Path

import generate_benchmark


def test_generate_dataset_writes_categorized_examples(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "hallucination_benchmark.json"
    monkeypatch.setattr(generate_benchmark, "Path", lambda _: target)

    generate_benchmark.generate_dataset()

    payload = json.loads(target.read_text(encoding="utf-8"))
    examples = payload["examples"]

    assert examples
    assert len({example["id"] for example in examples}) == len(examples)
    assert all(
        {
            "id",
            "category",
            "severity",
            "prompt",
            "response",
            "expected_hallucinations",
            "ground_truth",
        }
        <= set(example)
        for example in examples
    )

    categories = {example["category"] for example in examples}
    assert {
        "fabricated_verse",
        "fabricated_citation",
        "scholar_position_error",
        "temporal_confusion",
        "unsupported_claim",
    } <= categories
