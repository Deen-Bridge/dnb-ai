"""Arabic-language benchmark generation and evaluation utilities.

The benchmark is dependency-light by default so it can run in CI. Production
runs can inject an Arabic grammar checker and a model perplexity scorer through
the callables accepted by :class:`ArabicBenchmarkEvaluator`.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "eval" / "arabic_language_benchmark.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "eval" / "arabic_benchmark_report.json"

ARABIC_RE = re.compile(r"[\u0600-\u06ff]")
ARABIC_TOKEN_RE = re.compile(r"[\u0621-\u063a\u0641-\u064a]+|[0-9]+", re.UNICODE)
DIACRITIC_RE = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")
TATWEEL_RE = re.compile("ـ")

DIALECTS = ("msa", "classical", "egyptian", "levantine", "gulf", "maghrebi")
FORMALITY_LEVELS = ("formal", "neutral", "colloquial")
DIFFICULTIES = ("easy", "medium", "hard")
TASKS = (
    "comprehension",
    "generation",
    "diacritics",
    "grammar",
    "terminology",
    "script_handling",
    "cultural_context",
)

_QURAN_SAMPLES = (
    ("إِنَّ مَعَ الْعُسْرِ يُسْرًا", "إن مع العسر يسرا", "الشرح 94:6"),
    ("قُلْ هُوَ اللَّهُ أَحَدٌ", "قل هو الله أحد", "الإخلاص 112:1"),
    ("الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ", "الحمد لله رب العالمين", "الفاتحة 1:2"),
    ("مَالِكِ يَوْمِ الدِّينِ", "مالك يوم الدين", "الفاتحة 1:4"),
    ("إِيَّاكَ نَعْبُدُ وَإِيَّاكَ نَسْتَعِينُ", "إياك نعبد وإياك نستعين", "الفاتحة 1:5"),
    ("وَقُلْ رَبِّ زِدْنِي عِلْمًا", "وقل رب زدني علما", "طه 20:114"),
    ("لَا إِكْرَاهَ فِي الدِّينِ", "لا إكراه في الدين", "البقرة 2:256"),
    ("إِنَّ اللَّهَ مَعَ الصَّابِرِينَ", "إن الله مع الصابرين", "البقرة 2:153"),
    ("وَاللَّهُ يُحِبُّ الْمُحْسِنِينَ", "والله يحب المحسنين", "آل عمران 3:134"),
    ("إِنَّمَا الْمُؤْمِنُونَ إِخْوَةٌ", "إنما المؤمنون إخوة", "الحجرات 49:10"),
)

_TERMS = (
    ("التوحيد", "إفراد الله تعالى بما يختص به من الربوبية والألوهية والأسماء والصفات"),
    ("الإسناد", "سلسلة الرواة الذين نقلوا الحديث بعضهم عن بعض"),
    ("المتن", "ألفاظ الحديث التي ينتهي إليها الإسناد"),
    ("القياس", "إلحاق فرع بأصل في حكم لعلة جامعة بينهما"),
    ("الإجماع", "اتفاق مجتهدي الأمة في عصر على حكم شرعي"),
    ("الزكاة", "حق مالي واجب في مال مخصوص بشروطه ولمستحقيه"),
    ("الوقف", "تحبيس الأصل وتسبيل المنفعة في وجه مشروع"),
    ("الطهارة", "رفع الحدث وإزالة النجاسة على الوجه المشروع"),
    ("الحديث الصحيح", "ما اتصل سنده بنقل العدل الضابط من غير شذوذ ولا علة"),
    ("مقاصد الشريعة", "المعاني والغايات التي راعتها الشريعة لتحقيق المصالح"),
)

_MSA_ITEMS = (
    ("ما الفكرة الرئيسة في العبارة: العلم النافع يبني المجتمعات؟", "العلم النافع يسهم في بناء المجتمعات", ("العلم", "بناء", "المجتمعات")),
    ("لماذا تعد القراءة اليومية عادة مفيدة؟", "لأنها توسع المعرفة وتنمي التفكير واللغة", ("المعرفة", "التفكير", "اللغة")),
    ("اشرح بإيجاز أهمية حفظ الماء.", "حفظ الماء يمنع الهدر ويحمي موردا ضروريا للحياة", ("الماء", "الهدر", "الحياة")),
    ("ما أثر الصدق في العلاقات الاجتماعية؟", "يبني الصدق الثقة ويقوي العلاقات بين الناس", ("الثقة", "العلاقات")),
    ("لخص معنى التعاون في جملة واحدة.", "التعاون اشتراك الناس في العمل لتحقيق منفعة مشتركة", ("العمل", "منفعة", "مشتركة")),
    ("كيف يساعد تنظيم الوقت الطالب؟", "يساعده على إنجاز واجباته وتحقيق أهدافه بكفاءة", ("إنجاز", "أهداف", "كفاءة")),
    ("ما المقصود بالتفكير النقدي؟", "تحليل المعلومات والأدلة قبل قبول النتائج أو رفضها", ("تحليل", "المعلومات", "الأدلة")),
    ("بيّن فائدة الحوار الهادئ عند الاختلاف.", "يساعد الحوار الهادئ على الفهم وحل الخلاف باحترام", ("الفهم", "الخلاف", "احترام")),
    ("ما العلاقة بين التعليم والتنمية؟", "يرفع التعليم مهارات الناس ويدعم التنمية المستدامة", ("التعليم", "مهارات", "التنمية")),
    ("لماذا ينبغي التحقق من الأخبار قبل نشرها؟", "لتجنب نشر المعلومات الكاذبة والإضرار بالناس", ("التحقق", "الكاذبة", "الناس")),
)

_DIALECT_PROMPTS = {
    "egyptian": "حوّل المعنى إلى العربية الفصحى: إزيك؟ أنا عايز أعرف الميعاد.",
    "levantine": "حوّل المعنى إلى العربية الفصحى: شو الوقت وهلّق فينا نبلّش؟",
    "gulf": "حوّل المعنى إلى العربية الفصحى: شلونك؟ أبي أعرف وين المكان.",
    "maghrebi": "حوّل المعنى إلى العربية الفصحى: واش نقدر نعرف وقتاش نبداو؟",
}
_DIALECT_REFERENCES = {
    "egyptian": "كيف حالك؟ أريد أن أعرف الموعد.",
    "levantine": "ما الوقت؟ وهل يمكننا أن نبدأ الآن؟",
    "gulf": "كيف حالك؟ أريد أن أعرف أين المكان.",
    "maghrebi": "هل أستطيع أن أعرف متى نبدأ؟",
}


@dataclass(frozen=True)
class BenchmarkCase:
    id: str
    variety: str
    dialect: str
    formality: str
    difficulty: str
    task: str
    prompt: str
    reference_answer: str
    required_terms: tuple[str, ...] = ()
    forbidden_terms: tuple[str, ...] = ()
    source: str | None = None
    requires_diacritics: bool = False
    cultural_notes: str | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> BenchmarkCase:
        return cls(
            id=str(value["id"]),
            variety=str(value["variety"]),
            dialect=str(value["dialect"]),
            formality=str(value["formality"]),
            difficulty=str(value["difficulty"]),
            task=str(value["task"]),
            prompt=str(value["prompt"]),
            reference_answer=str(value["reference_answer"]),
            required_terms=tuple(str(item) for item in value.get("required_terms", ())),
            forbidden_terms=tuple(str(item) for item in value.get("forbidden_terms", ())),
            source=str(value["source"]) if value.get("source") is not None else None,
            requires_diacritics=bool(value.get("requires_diacritics", False)),
            cultural_notes=(str(value["cultural_notes"]) if value.get("cultural_notes") is not None else None),
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["required_terms"] = list(self.required_terms)
        value["forbidden_terms"] = list(self.forbidden_terms)
        return value


@dataclass(frozen=True)
class HumanRating:
    case_id: str
    evaluator_id: str
    fluency: float
    grammar: float
    naturalness: float
    appropriateness: float
    terminology: float
    comments: str = ""

    def __post_init__(self) -> None:
        for field in ("fluency", "grammar", "naturalness", "appropriateness", "terminology"):
            value = getattr(self, field)
            if not 1.0 <= value <= 5.0:
                raise ValueError(f"{field} must be between 1 and 5")


def strip_diacritics(text: str) -> str:
    return TATWEEL_RE.sub("", DIACRITIC_RE.sub("", unicodedata.normalize("NFC", text)))


def normalize_arabic(text: str, *, keep_diacritics: bool = False) -> str:
    normalized = unicodedata.normalize("NFC", text).strip()
    if not keep_diacritics:
        normalized = strip_diacritics(normalized)
    normalized = normalized.translate(str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي"}))
    return " ".join(normalized.split())


def tokenize_arabic(text: str) -> list[str]:
    return ARABIC_TOKEN_RE.findall(normalize_arabic(text).lower())


def corpus_bleu(reference: str, candidate: str, max_order: int = 4) -> float:
    """Calculate smoothed single-reference BLEU in the range 0..1."""
    reference_tokens = tokenize_arabic(reference)
    candidate_tokens = tokenize_arabic(candidate)
    if not reference_tokens or not candidate_tokens:
        return 0.0
    precisions: list[float] = []
    for order in range(1, max_order + 1):
        candidate_ngrams = Counter(tuple(candidate_tokens[i : i + order]) for i in range(len(candidate_tokens) - order + 1))
        reference_ngrams = Counter(tuple(reference_tokens[i : i + order]) for i in range(len(reference_tokens) - order + 1))
        overlap = sum((candidate_ngrams & reference_ngrams).values())
        total = sum(candidate_ngrams.values())
        precisions.append((overlap + 1.0) / (total + 1.0))
    brevity_penalty = 1.0 if len(candidate_tokens) > len(reference_tokens) else math.exp(1 - len(reference_tokens) / len(candidate_tokens))
    return brevity_penalty * math.exp(sum(math.log(value) for value in precisions) / max_order)


def diacritical_accuracy(reference: str, candidate: str) -> float:
    """Compare base letters and their following combining marks."""
    reference = unicodedata.normalize("NFD", reference)
    candidate = unicodedata.normalize("NFD", candidate)

    def units(text: str) -> list[tuple[str, str]]:
        result: list[tuple[str, str]] = []
        for character in text:
            if unicodedata.combining(character) and result:
                base, marks = result[-1]
                result[-1] = (base, marks + character)
            elif ARABIC_RE.match(character):
                result.append((character, ""))
        return result

    reference_units = units(reference)
    candidate_units = units(candidate)
    if not reference_units:
        return 1.0
    matches = sum(1 for expected, actual in zip(reference_units, candidate_units, strict=False) if expected == actual)
    return matches / max(len(reference_units), len(candidate_units), 1)


def script_handling_score(text: str) -> float:
    visible = [character for character in text if not character.isspace()]
    if not visible:
        return 0.0
    arabic = sum(1 for character in visible if ARABIC_RE.match(character))
    control_penalty = sum(1 for character in visible if unicodedata.category(character) in {"Cc", "Cs"})
    replacement_penalty = text.count("�") + text.count("?")
    return max(0.0, min(1.0, (arabic - control_penalty - replacement_penalty) / len(visible)))


def _contains(text: str, term: str) -> bool:
    return normalize_arabic(term).lower() in normalize_arabic(text).lower()


def _default_grammar_errors(text: str) -> int:
    """Conservative fallback checks; inject an Arabic NLP checker in full runs."""
    errors = 0
    errors += len(re.findall(r"\s+[،؛؟,.]", text))
    errors += len(re.findall(r"[،؛؟,.]{2,}", text))
    errors += len(re.findall(r"\b(في|من|إلى|على|عن)\s+\1\b", text))
    if text.strip() and text.strip()[-1] not in ".؟!؛":
        errors += 1
    return errors


def generate_cases(count: int = 420) -> list[BenchmarkCase]:
    """Generate a deterministic, balanced benchmark containing at least 400 cases."""
    if count < 400:
        raise ValueError("Arabic benchmark must contain at least 400 cases")
    cases: list[BenchmarkCase] = []
    for index in range(count):
        task = TASKS[index % len(TASKS)]
        difficulty = DIFFICULTIES[(index // len(TASKS)) % len(DIFFICULTIES)]
        formality = FORMALITY_LEVELS[(index // (len(TASKS) * len(DIFFICULTIES))) % len(FORMALITY_LEVELS)]
        suffix = index + 1
        if task == "diacritics":
            vocalized, plain, source = _QURAN_SAMPLES[index % len(_QURAN_SAMPLES)]
            case = BenchmarkCase(
                id=f"arabic-{suffix:04d}",
                variety="classical",
                dialect="classical",
                formality="formal",
                difficulty=difficulty,
                task=task,
                prompt=f"اكتب النص القرآني الآتي مضبوطا بالشكل كما في المصحف: {plain}",
                reference_answer=vocalized,
                source=source,
                requires_diacritics=True,
                cultural_notes="يجب نقل النص المقدس بدقة ومن دون إعادة صياغة.",
            )
        elif task == "terminology":
            term, definition = _TERMS[index % len(_TERMS)]
            case = BenchmarkCase(
                id=f"arabic-{suffix:04d}",
                variety="classical",
                dialect="classical",
                formality="formal",
                difficulty=difficulty,
                task=task,
                prompt=f"عرّف المصطلح الإسلامي الآتي تعريفا دقيقا: {term}.",
                reference_answer=f"{term}: {definition}.",
                required_terms=(term,),
                source="مصطلح إسلامي قياسي",
            )
        elif task == "script_handling":
            dialect = tuple(_DIALECT_PROMPTS)[index % len(_DIALECT_PROMPTS)]
            case = BenchmarkCase(
                id=f"arabic-{suffix:04d}",
                variety="dialect",
                dialect=dialect,
                formality="colloquial",
                difficulty=difficulty,
                task=task,
                prompt=_DIALECT_PROMPTS[dialect],
                reference_answer=_DIALECT_REFERENCES[dialect],
                cultural_notes="حافظ على المعنى وتجنب السخرية من اللهجة.",
            )
        else:
            prompt, answer, terms = _MSA_ITEMS[index % len(_MSA_ITEMS)]
            case = BenchmarkCase(
                id=f"arabic-{suffix:04d}",
                variety="msa",
                dialect="msa",
                formality=formality,
                difficulty=difficulty,
                task=task,
                prompt=prompt,
                reference_answer=answer + ".",
                required_terms=terms if task in {"comprehension", "cultural_context"} else (),
                cultural_notes=("قيّم الاحترام والسياق الاجتماعي العربي." if task == "cultural_context" else None),
            )
        cases.append(case)
    return cases


def validate_cases(cases: Sequence[BenchmarkCase]) -> list[str]:
    errors: list[str] = []
    ids: set[str] = set()
    for case in cases:
        if case.id in ids:
            errors.append(f"Duplicate id: {case.id}")
        ids.add(case.id)
        if case.difficulty not in DIFFICULTIES:
            errors.append(f"{case.id}: invalid difficulty")
        if case.formality not in FORMALITY_LEVELS:
            errors.append(f"{case.id}: invalid formality")
        if case.task not in TASKS:
            errors.append(f"{case.id}: invalid task")
        if case.dialect not in DIALECTS:
            errors.append(f"{case.id}: invalid dialect")
        if not ARABIC_RE.search(case.prompt) or not ARABIC_RE.search(case.reference_answer):
            errors.append(f"{case.id}: prompt and reference must contain Arabic")
        if case.requires_diacritics and not DIACRITIC_RE.search(case.reference_answer):
            errors.append(f"{case.id}: diacritical case has no reference diacritics")
    return errors


def write_dataset(path: Path = DEFAULT_DATASET, count: int = 420) -> list[BenchmarkCase]:
    cases = generate_cases(count)
    errors = validate_cases(cases)
    if errors:
        raise ValueError("Invalid generated benchmark: " + "; ".join(errors))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case.to_dict(), ensure_ascii=False) + "\n")
    return cases


def load_dataset(path: Path = DEFAULT_DATASET) -> list[BenchmarkCase]:
    if not path.exists():
        return generate_cases()
    with path.open(encoding="utf-8") as handle:
        if path.suffix == ".json":
            raw = json.load(handle)
        else:
            raw = [json.loads(line) for line in handle if line.strip()]
    if not isinstance(raw, list):
        raise ValueError("Benchmark dataset must be a JSON array or JSONL records")
    cases = [BenchmarkCase.from_dict(value) for value in raw]
    errors = validate_cases(cases)
    if errors:
        raise ValueError("Invalid benchmark dataset: " + "; ".join(errors))
    return cases


class ArabicBenchmarkEvaluator:
    def __init__(
        self,
        grammar_checker: Callable[[str], int] | None = None,
        perplexity_scorer: Callable[[str], float] | None = None,
    ) -> None:
        self.grammar_checker = grammar_checker or _default_grammar_errors
        self.perplexity_scorer = perplexity_scorer

    def score(self, case: BenchmarkCase, response: str) -> dict[str, Any]:
        tokens = tokenize_arabic(response)
        required_hits = sum(_contains(response, term) for term in case.required_terms)
        terminology = required_hits / len(case.required_terms) if case.required_terms else 1.0
        forbidden_pass = not any(_contains(response, term) for term in case.forbidden_terms)
        bleu = corpus_bleu(case.reference_answer, response)
        grammar_errors = max(0, self.grammar_checker(response))
        grammar_error_rate = grammar_errors / max(len(tokens), 1)
        diacritics = diacritical_accuracy(case.reference_answer, response) if case.requires_diacritics else None
        comprehension = 0.65 * terminology + 0.35 * bleu
        appropriateness = 1.0 if forbidden_pass and script_handling_score(response) >= 0.5 else 0.0
        return {
            "id": case.id,
            "task": case.task,
            "variety": case.variety,
            "dialect": case.dialect,
            "difficulty": case.difficulty,
            "bleu": round(bleu, 6),
            "comprehension_accuracy": round(comprehension, 6),
            "grammar_errors": grammar_errors,
            "grammar_error_rate": round(grammar_error_rate, 6),
            "diacritical_accuracy": round(diacritics, 6) if diacritics is not None else None,
            "terminology_accuracy": round(terminology, 6),
            "script_handling": round(script_handling_score(response), 6),
            "cultural_appropriateness": appropriateness,
            "perplexity": self.perplexity_scorer(response) if self.perplexity_scorer else None,
        }

    def evaluate(
        self,
        cases: Sequence[BenchmarkCase],
        responses: Mapping[str, str],
        human_ratings: Iterable[HumanRating] = (),
        english_baseline_accuracy: float | None = None,
    ) -> dict[str, Any]:
        missing = [case.id for case in cases if case.id not in responses]
        if missing:
            raise ValueError(f"Missing responses for {len(missing)} cases; first missing id: {missing[0]}")
        results = [self.score(case, responses[case.id]) for case in cases]
        by_variety: dict[str, list[dict[str, Any]]] = defaultdict(list)
        by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for result in results:
            by_variety[result["variety"]].append(result)
            by_task[result["task"]].append(result)

        def average(key: str, rows: Sequence[Mapping[str, Any]]) -> float | None:
            values = [float(row[key]) for row in rows if row.get(key) is not None]
            return round(sum(values) / len(values), 6) if values else None

        ratings = list(human_ratings)
        human = {
            field: round(sum(getattr(rating, field) for rating in ratings) / len(ratings), 4) if ratings else None
            for field in ("fluency", "grammar", "naturalness", "appropriateness", "terminology")
        }
        overall = {
            key: average(key, results)
            for key in (
                "bleu",
                "comprehension_accuracy",
                "grammar_error_rate",
                "diacritical_accuracy",
                "terminology_accuracy",
                "script_handling",
                "cultural_appropriateness",
                "perplexity",
            )
        }
        baseline_ratio = None
        if english_baseline_accuracy is not None:
            if not 0 < english_baseline_accuracy <= 1:
                raise ValueError("English baseline accuracy must be in the range (0, 1]")
            overall_accuracy = overall["comprehension_accuracy"]
            baseline_ratio = round(float(overall_accuracy) / english_baseline_accuracy, 6)
        criteria = {
            "comprehension_vs_english_baseline_over_90_percent": baseline_ratio is not None and baseline_ratio > 0.90,
            "human_fluency_over_4_2": human["fluency"] is not None and human["fluency"] > 4.2,
            "quranic_diacritical_accuracy_over_98_percent": (
                overall["diacritical_accuracy"] is not None and overall["diacritical_accuracy"] > 0.98
            ),
            "grammar_error_rate_under_5_percent": (
                overall["grammar_error_rate"] is not None and overall["grammar_error_rate"] < 0.05
            ),
            "terminology_accuracy_over_92_percent": (
                overall["terminology_accuracy"] is not None and overall["terminology_accuracy"] > 0.92
            ),
            "cultural_appropriateness_over_88_percent": (
                overall["cultural_appropriateness"] is not None and overall["cultural_appropriateness"] > 0.88
            ),
        }
        return {
            "total_cases": len(cases),
            "overall": overall,
            "arabic_to_english_comprehension_ratio": baseline_ratio,
            "human_evaluation": human,
            "by_variety": {
                name: {"count": len(rows), "comprehension_accuracy": average("comprehension_accuracy", rows), "bleu": average("bleu", rows)}
                for name, rows in sorted(by_variety.items())
            },
            "by_task": {
                name: {"count": len(rows), "primary_score": average(_primary_metric(name), rows)}
                for name, rows in sorted(by_task.items())
            },
            "success_criteria": criteria,
            "results": results,
        }


def _primary_metric(task: str) -> str:
    return {
        "diacritics": "diacritical_accuracy",
        "grammar": "grammar_error_rate",
        "terminology": "terminology_accuracy",
        "script_handling": "script_handling",
        "cultural_context": "cultural_appropriateness",
    }.get(task, "comprehension_accuracy")


def compare_models(reports: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Create a deterministic leaderboard for this system and specialized models."""
    leaderboard = []
    for model_name, report in reports.items():
        overall = report.get("overall", {})
        quality_values = [
            float(overall[key])
            for key in ("comprehension_accuracy", "bleu", "diacritical_accuracy", "terminology_accuracy", "script_handling")
            if overall.get(key) is not None
        ]
        grammar_rate = float(overall.get("grammar_error_rate", 1.0))
        aggregate = (sum(quality_values) + max(0.0, 1.0 - grammar_rate)) / (len(quality_values) + 1)
        leaderboard.append({"model": model_name, "aggregate_score": round(aggregate, 6), "overall": overall})
    return sorted(leaderboard, key=lambda row: (-row["aggregate_score"], row["model"]))


def load_responses(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict) or not all(isinstance(key, str) and isinstance(value, str) for key, value in raw.items()):
        raise ValueError("Responses file must be a JSON object mapping case ids to response strings")
    return raw


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or evaluate the Deen Bridge Arabic benchmark")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--build", action="store_true", help="Write the deterministic 420-case JSONL dataset")
    parser.add_argument("--responses", type=Path, help="JSON object mapping case ids to model responses")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--english-baseline", type=float)
    args = parser.parse_args()

    cases = write_dataset(args.dataset) if args.build else load_dataset(args.dataset)
    print(f"Loaded {len(cases)} valid Arabic benchmark cases.")
    if args.responses:
        report = ArabicBenchmarkEvaluator().evaluate(
            cases,
            load_responses(args.responses),
            english_baseline_accuracy=args.english_baseline,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote evaluation report to {args.output}")


if __name__ == "__main__":
    main()
