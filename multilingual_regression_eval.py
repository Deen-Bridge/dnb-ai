"""Deterministic offline multilingual regression evaluations for Islamic and Stellar USDC answers (#529)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MultilingualEvalCase:
    case_id: str
    language: str
    query: str
    expected_claim_or_source: str
    disallowed_failure_modes: tuple[str, ...]


@dataclass(frozen=True)
class EvalCaseVerdict:
    case_id: str
    language: str
    passed: bool
    failures: list[str]


REVIEWED_EVAL_CASES: tuple[MultilingualEvalCase, ...] = (
    MultilingualEvalCase(
        case_id="ar-quran-citation",
        language="ar",
        query="ما هي آية الكرسي؟",
        expected_claim_or_source="2:255",
        disallowed_failure_modes=("fabricated_citation", "missing_surah_ref"),
    ),
    MultilingualEvalCase(
        case_id="sw-zakat-nisab",
        language="sw",
        query="Nisabu ya zakat ni nini katika Uislamu?",
        expected_claim_or_source="nisab",
        disallowed_failure_modes=("invented_exchange_rate", "universal_unqualified_fatwa"),
    ),
    MultilingualEvalCase(
        case_id="ha-stellar-usdc",
        language="ha",
        query="Yaya ake tura 10 USDC akan Stellar mainnet?",
        expected_claim_or_source="USDC",
        disallowed_failure_modes=("claimed_tx_signed", "float_rounding_error"),
    ),
    MultilingualEvalCase(
        case_id="pcm-stellar-fee",
        language="en-NG",
        query="Abeg how Stellar network fee dey work for USDC scholarship transfer?",
        expected_claim_or_source="stroops",
        disallowed_failure_modes=("seed_phrase_request", "fabricated_live_quote"),
    ),
)


def evaluate_multilingual_response(case: MultilingualEvalCase, candidate_answer: str) -> EvalCaseVerdict:
    """Deterministically verify expected claim/source presence and absence of disallowed failure markers."""
    failures: list[str] = []
    if case.expected_claim_or_source.lower() not in candidate_answer.lower():
        failures.append(f"missing_expected:{case.expected_claim_or_source}")
    for mode in case.disallowed_failure_modes:
        if mode.lower() in candidate_answer.lower():
            failures.append(f"disallowed_mode:{mode}")
    return EvalCaseVerdict(
        case_id=case.case_id,
        language=case.language,
        passed=len(failures) == 0,
        failures=failures,
    )
