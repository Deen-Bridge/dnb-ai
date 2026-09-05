"""Evaluate answer quality and parity across supported languages.

The benchmark consumes two JSONL files:

Dataset records::

    {
      "id": "aqeedah-001",
      "translations": {
        "en": {"question": "...", "reference_answer": "..."},
        "ar": {"question": "...", "reference_answer": "..."}
      },
      "terms": {
        "en": ["Tawhid"],
        "ar": ["التوحيد"]
      }
    }

Prediction records::

    {"id": "aqeedah-001", "language": "en", "answer": "...", "accurate": true}

Predictions may include a precomputed ``comet`` score. A COMET scorer can also be
provided through the Python API. Human evaluations are optional JSONL records
with ``id``, ``language``, ``evaluator_id``, and integer scores from one to five
for cultural appropriateness, terminology, tone, and source appropriateness.

Semantic scoring deliberately uses an injected multilingual scorer rather than
a monolingual lexical approximation. Production runs should inject the same
versioned multilingual embedding or entailment model for every language pair.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

SUPPORTED_LANGUAGES = ("en", "ar", "ur", "tr", "ms", "fr")
MINIMUM_QUESTION_COUNT = 200
MINIMUM_LANGUAGE_COUNT = 6

SemanticScorer = Callable[[str, str], float]
CometScorer = Callable[[str, str, str], float]

_WORD_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)
_ARABIC_RANGES = (
    (0x0600, 0x06FF),
    (0x0750, 0x077F),
    (0x0870, 0x089F),
    (0x08A0, 0x08FF),
    (0xFB50, 0xFDFF),
    (0xFE70, 0xFEFF),
)


class BenchmarkValidationError(ValueError):
    """Raised when benchmark input does not satisfy the parallel-set schema."""


@dataclass(frozen=True)
class Translation:
    question: str
    reference_answer: str


@dataclass(frozen=True)
class BenchmarkItem:
    item_id: str
    translations: Mapping[str, Translation]
    terms: Mapping[str, tuple[str, ...]]


@dataclass(frozen=True)
class Prediction:
    item_id: str
    language: str
    answer: str
    accurate: bool | None = None
    comet: float | None = None


@dataclass(frozen=True)
class HumanEvaluation:
    item_id: str
    language: str
    evaluator_id: str
    cultural_appropriateness: int
    terminology: int
    tone: int
    source_appropriateness: int

    @property
    def overall(self) -> float:
        return (
            self.cultural_appropriateness
            + self.terminology
            + self.tone
            + self.source_appropriateness
        ) / 4


def _required_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkValidationError(f"{field} must be a non-empty string")
    return value


def _optional_unit_score(value: Any, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BenchmarkValidationError(f"{field} must be a number between 0 and 1")
    score = float(value)
    if not 0 <= score <= 1:
        raise BenchmarkValidationError(f"{field} must be between 0 and 1")
    return score


def parse_item(raw: Mapping[str, Any]) -> BenchmarkItem:
    """Parse and validate one parallel benchmark item."""
    item_id = _required_string(raw.get("id"), "id")
    translations_raw = raw.get("translations")
    if not isinstance(translations_raw, dict) or not translations_raw:
        raise BenchmarkValidationError(f"{item_id}: translations must be a non-empty object")

    translations: dict[str, Translation] = {}
    for language, value in translations_raw.items():
        if not isinstance(language, str) or not isinstance(value, dict):
            raise BenchmarkValidationError(f"{item_id}: invalid translation entry")
        translations[language] = Translation(
            question=_required_string(value.get("question"), f"{item_id}.{language}.question"),
            reference_answer=_required_string(
                value.get("reference_answer"),
                f"{item_id}.{language}.reference_answer",
            ),
        )

    terms_raw = raw.get("terms", {})
    if not isinstance(terms_raw, dict):
        raise BenchmarkValidationError(f"{item_id}: terms must be an object")
    terms: dict[str, tuple[str, ...]] = {}
    for language, values in terms_raw.items():
        if not isinstance(language, str) or not isinstance(values, list):
            raise BenchmarkValidationError(f"{item_id}: terms.{language} must be an array")
        parsed_terms = tuple(_required_string(value, f"{item_id}.terms.{language}") for value in values)
        terms[language] = parsed_terms

    return BenchmarkItem(item_id=item_id, translations=translations, terms=terms)


def parse_prediction(raw: Mapping[str, Any]) -> Prediction:
    """Parse one generated answer record."""
    accurate_raw = raw.get("accurate")
    if accurate_raw is not None and not isinstance(accurate_raw, bool):
        raise BenchmarkValidationError("prediction accurate must be a boolean when provided")
    return Prediction(
        item_id=_required_string(raw.get("id"), "prediction.id"),
        language=_required_string(raw.get("language"), "prediction.language"),
        answer=_required_string(raw.get("answer"), "prediction.answer"),
        accurate=accurate_raw,
        comet=_optional_unit_score(raw.get("comet"), "prediction.comet"),
    )


def _human_score(raw: Mapping[str, Any], field: str) -> int:
    value = raw.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
        raise BenchmarkValidationError(f"human evaluation {field} must be an integer from 1 to 5")
    return value


def parse_human_evaluation(raw: Mapping[str, Any]) -> HumanEvaluation:
    """Parse one native-speaker evaluation using the four-part scoring rubric."""
    return HumanEvaluation(
        item_id=_required_string(raw.get("id"), "human.id"),
        language=_required_string(raw.get("language"), "human.language"),
        evaluator_id=_required_string(raw.get("evaluator_id"), "human.evaluator_id"),
        cultural_appropriateness=_human_score(raw, "cultural_appropriateness"),
        terminology=_human_score(raw, "terminology"),
        tone=_human_score(raw, "tone"),
        source_appropriateness=_human_score(raw, "source_appropriateness"),
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSON objects from a UTF-8 JSONL file with useful line errors."""
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BenchmarkValidationError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise BenchmarkValidationError(f"{path}:{line_number}: record must be a JSON object")
            records.append(value)
    return records


def load_dataset(path: Path) -> list[BenchmarkItem]:
    return [parse_item(record) for record in load_jsonl(path)]


def load_predictions(path: Path) -> list[Prediction]:
    return [parse_prediction(record) for record in load_jsonl(path)]


def load_human_evaluations(path: Path) -> list[HumanEvaluation]:
    return [parse_human_evaluation(record) for record in load_jsonl(path)]


def validate_parallel_dataset(
    items: Sequence[BenchmarkItem],
    minimum_questions: int = MINIMUM_QUESTION_COUNT,
    minimum_languages: int = MINIMUM_LANGUAGE_COUNT,
) -> tuple[str, ...]:
    """Validate size, uniqueness, and complete language coverage.

    Every selected benchmark language must occur on every item. This prevents a
    favorable aggregate from hiding missing or disproportionately small language
    samples.
    """
    if len(items) < minimum_questions:
        raise BenchmarkValidationError(
            f"benchmark requires at least {minimum_questions} questions; found {len(items)}"
        )
    identifiers = [item.item_id for item in items]
    duplicates = sorted(item_id for item_id, count in Counter(identifiers).items() if count > 1)
    if duplicates:
        raise BenchmarkValidationError(f"duplicate question ids: {', '.join(duplicates)}")

    language_sets = [set(item.translations) for item in items]
    common_languages = set.intersection(*language_sets) if language_sets else set()
    if len(common_languages) < minimum_languages:
        raise BenchmarkValidationError(
            f"benchmark requires {minimum_languages} languages on every question; "
            f"found {len(common_languages)} ({', '.join(sorted(common_languages))})"
        )
    if "en" not in common_languages:
        raise BenchmarkValidationError("English is required as the performance baseline")
    return tuple(language for language in SUPPORTED_LANGUAGES if language in common_languages) + tuple(
        sorted(common_languages.difference(SUPPORTED_LANGUAGES))
    )


def tokenize(text: str) -> list[str]:
    """Unicode-aware tokenization suitable for the benchmark's supported scripts."""
    return [token.casefold() for token in _WORD_RE.findall(unicodedata.normalize("NFC", text))]


def bleu_score(reference: str, candidate: str, maximum_order: int = 4) -> float:
    """Calculate a smoothed sentence BLEU score in the range zero to one."""
    if maximum_order < 1:
        raise ValueError("maximum_order must be positive")
    reference_tokens = tokenize(reference)
    candidate_tokens = tokenize(candidate)
    if not candidate_tokens or not reference_tokens:
        return 0.0

    precisions: list[float] = []
    for order in range(1, maximum_order + 1):
        candidate_ngrams = Counter(
            tuple(candidate_tokens[index : index + order])
            for index in range(max(0, len(candidate_tokens) - order + 1))
        )
        reference_ngrams = Counter(
            tuple(reference_tokens[index : index + order])
            for index in range(max(0, len(reference_tokens) - order + 1))
        )
        overlap = sum(min(count, reference_ngrams[ngram]) for ngram, count in candidate_ngrams.items())
        total = sum(candidate_ngrams.values())
        precisions.append((overlap + 1) / (total + 1))

    brevity_penalty = (
        1.0
        if len(candidate_tokens) >= len(reference_tokens)
        else math.exp(1 - len(reference_tokens) / len(candidate_tokens))
    )
    return brevity_penalty * math.exp(sum(math.log(precision) for precision in precisions) / maximum_order)


def _is_arabic_character(character: str) -> bool:
    codepoint = ord(character)
    return any(start <= codepoint <= end for start, end in _ARABIC_RANGES)


def script_is_valid(text: str, language: str) -> bool:
    """Check Unicode integrity and expected script without assuming a font engine.

    Rendering correctness here means the service preserved valid normalized text
    in the language's expected script. Visual glyph shaping remains a client/font
    integration concern and should be included in native-speaker review.
    """
    if not text or "\ufffd" in text or unicodedata.normalize("NFC", text) != text:
        return False
    for character in text:
        category = unicodedata.category(character)
        if category in {"Cs", "Co", "Cn"}:
            return False
        if category == "Cc" and character not in "\n\r\t":
            return False

    letters = [character for character in text if unicodedata.category(character).startswith("L")]
    if not letters:
        return False
    if language in {"ar", "ur"}:
        return sum(_is_arabic_character(character) for character in letters) / len(letters) >= 0.6
    return sum("LATIN" in unicodedata.name(character, "") for character in letters) / len(letters) >= 0.6


def terminology_accuracy(answer: str, expected_terms: Sequence[str]) -> float | None:
    """Return the proportion of required localized Islamic terms present."""
    if not expected_terms:
        return None
    normalized_answer = " ".join(tokenize(answer))
    matched = 0
    for term in expected_terms:
        normalized_term = " ".join(tokenize(term))
        if normalized_term and normalized_term in normalized_answer:
            matched += 1
    return matched / len(expected_terms)


def _mean(values: Iterable[float]) -> float | None:
    materialized = list(values)
    return sum(materialized) / len(materialized) if materialized else None


def _round_optional(value: float | None) -> float | None:
    return round(value, 4) if value is not None else None


def evaluate(
    items: Sequence[BenchmarkItem],
    predictions: Sequence[Prediction],
    human_evaluations: Sequence[HumanEvaluation] = (),
    semantic_scorer: SemanticScorer | None = None,
    comet_scorer: CometScorer | None = None,
    minimum_questions: int = MINIMUM_QUESTION_COUNT,
    minimum_languages: int = MINIMUM_LANGUAGE_COUNT,
) -> dict[str, Any]:
    """Evaluate multilingual predictions and return a JSON-serializable report."""
    languages = validate_parallel_dataset(items, minimum_questions, minimum_languages)
    item_map = {item.item_id: item for item in items}
    prediction_map: dict[tuple[str, str], Prediction] = {}
    for prediction in predictions:
        key = (prediction.item_id, prediction.language)
        if prediction.item_id not in item_map:
            raise BenchmarkValidationError(f"prediction references unknown id {prediction.item_id}")
        if prediction.language not in languages:
            raise BenchmarkValidationError(f"prediction uses unsupported dataset language {prediction.language}")
        if key in prediction_map:
            raise BenchmarkValidationError(
                f"duplicate prediction for {prediction.item_id}/{prediction.language}"
            )
        prediction_map[key] = prediction

    missing = [
        f"{item.item_id}/{language}"
        for item in items
        for language in languages
        if (item.item_id, language) not in prediction_map
    ]
    if missing:
        preview = ", ".join(missing[:10])
        raise BenchmarkValidationError(f"missing {len(missing)} predictions: {preview}")

    valid_human_keys = set(prediction_map)
    seen_human: set[tuple[str, str, str]] = set()
    for evaluation in human_evaluations:
        key = (evaluation.item_id, evaluation.language)
        evaluator_key = (evaluation.item_id, evaluation.language, evaluation.evaluator_id)
        if key not in valid_human_keys:
            raise BenchmarkValidationError(
                f"human evaluation references unknown prediction {evaluation.item_id}/{evaluation.language}"
            )
        if evaluator_key in seen_human:
            raise BenchmarkValidationError("duplicate human evaluation by the same evaluator")
        seen_human.add(evaluator_key)

    human_by_language: dict[str, list[HumanEvaluation]] = defaultdict(list)
    for evaluation in human_evaluations:
        human_by_language[evaluation.language].append(evaluation)

    per_language: dict[str, dict[str, Any]] = {}
    accuracy_values: dict[str, float] = {}
    all_semantic_scores: list[float] = []
    all_term_scores: list[float] = []
    all_human_scores: list[float] = []
    all_comet_scores: list[float] = []
    script_checks = 0
    script_passes = 0

    for language in languages:
        bleu_values: list[float] = []
        semantic_values: list[float] = []
        term_values: list[float] = []
        comet_values: list[float] = []
        accuracy_flags: list[bool] = []
        language_script_passes = 0

        for item in items:
            translation = item.translations[language]
            prediction = prediction_map[(item.item_id, language)]
            bleu_values.append(bleu_score(translation.reference_answer, prediction.answer))
            valid_script = script_is_valid(prediction.answer, language)
            language_script_passes += int(valid_script)
            script_checks += 1
            script_passes += int(valid_script)

            if semantic_scorer is not None:
                score = float(semantic_scorer(translation.reference_answer, prediction.answer))
                if not 0 <= score <= 1:
                    raise BenchmarkValidationError("semantic scorer returned a value outside 0..1")
                semantic_values.append(score)
                all_semantic_scores.append(score)

            term_score = terminology_accuracy(prediction.answer, item.terms.get(language, ()))
            if term_score is not None:
                term_values.append(term_score)
                all_term_scores.append(term_score)

            comet_score = prediction.comet
            if comet_scorer is not None:
                comet_score = float(
                    comet_scorer(
                        item.translations["en"].question,
                        prediction.answer,
                        translation.reference_answer,
                    )
                )
                if not 0 <= comet_score <= 1:
                    raise BenchmarkValidationError("COMET scorer returned a value outside 0..1")
            if comet_score is not None:
                comet_values.append(comet_score)
                all_comet_scores.append(comet_score)
            if prediction.accurate is not None:
                accuracy_flags.append(prediction.accurate)

        human_values = [evaluation.overall for evaluation in human_by_language[language]]
        all_human_scores.extend(human_values)
        accuracy = _mean(float(value) for value in accuracy_flags)
        if accuracy is not None:
            accuracy_values[language] = accuracy
        per_language[language] = {
            "questions": len(items),
            "bleu": _round_optional(_mean(bleu_values)),
            "comet": _round_optional(_mean(comet_values)),
            "semantic_accuracy": _round_optional(_mean(semantic_values)),
            "islamic_term_accuracy": _round_optional(_mean(term_values)),
            "script_correctness": round(language_script_passes / len(items), 4),
            "human_cultural_appropriateness": _round_optional(_mean(human_values)),
            "human_evaluation_count": len(human_values),
            "accuracy": _round_optional(accuracy),
        }

    cross_lingual_scores: list[float] = []
    if semantic_scorer is not None:
        for item in items:
            for first_language, second_language in combinations(languages, 2):
                first_answer = prediction_map[(item.item_id, first_language)].answer
                second_answer = prediction_map[(item.item_id, second_language)].answer
                score = float(semantic_scorer(first_answer, second_answer))
                if not 0 <= score <= 1:
                    raise BenchmarkValidationError("semantic scorer returned a value outside 0..1")
                cross_lingual_scores.append(score)

    english_accuracy = accuracy_values.get("en")
    performance_gaps: dict[str, float | None] = {}
    for language in languages:
        language_accuracy = accuracy_values.get(language)
        gap = None
        if english_accuracy is not None and language_accuracy is not None:
            gap = english_accuracy - language_accuracy
        performance_gaps[language] = _round_optional(gap)

    cross_lingual = _mean(cross_lingual_scores)
    term_accuracy = _mean(all_term_scores)
    cultural_score = _mean(all_human_scores)
    script_correctness = script_passes / script_checks
    non_english_gaps = [
        gap for language, gap in performance_gaps.items() if language != "en" and gap is not None
    ]

    criteria = {
        "semantic_equivalence_above_88_percent": cross_lingual is not None and cross_lingual > 0.88,
        "per_language_accuracy_within_5_percent_of_english": bool(non_english_gaps)
        and all(abs(gap) <= 0.05 for gap in non_english_gaps),
        "cultural_appropriateness_above_4_of_5": cultural_score is not None and cultural_score > 4.0,
        "islamic_term_accuracy_above_92_percent": term_accuracy is not None and term_accuracy > 0.92,
        "script_rendering_correctness_100_percent": script_correctness == 1.0,
        "no_language_degradation_above_10_percent": bool(non_english_gaps)
        and all(gap <= 0.10 for gap in non_english_gaps),
    }

    return {
        "benchmark": {
            "question_count": len(items),
            "languages": list(languages),
            "language_count": len(languages),
            "prediction_count": len(predictions),
        },
        "aggregate": {
            "cross_lingual_semantic_equivalence": _round_optional(cross_lingual),
            "reference_semantic_accuracy": _round_optional(_mean(all_semantic_scores)),
            "islamic_term_accuracy": _round_optional(term_accuracy),
            "comet": _round_optional(_mean(all_comet_scores)),
            "cultural_appropriateness": _round_optional(cultural_score),
            "script_correctness": round(script_correctness, 4),
        },
        "per_language": per_language,
        "performance_gap_from_english": performance_gaps,
        "success_criteria": criteria,
        "passed": all(criteria.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Parallel benchmark JSONL")
    parser.add_argument("--predictions", type=Path, required=True, help="Generated answer JSONL")
    parser.add_argument("--human-evaluations", type=Path, help="Optional native-speaker evaluation JSONL")
    parser.add_argument("--output", type=Path, help="Write report JSON to this path instead of stdout")
    args = parser.parse_args()

    items = load_dataset(args.dataset)
    predictions = load_predictions(args.predictions)
    human_evaluations = (
        load_human_evaluations(args.human_evaluations) if args.human_evaluations else []
    )
    report = evaluate(items, predictions, human_evaluations)
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(serialized, encoding="utf-8")
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()
