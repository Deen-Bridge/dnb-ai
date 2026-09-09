"""Cross-session contradiction detection engine.

Compares candidate assertions against historical claim indexes and canonical
anchors to identify ruling contradictions, attribution mismatches, evidence
clashes, and numerical drift.
"""

from __future__ import annotations

import re

from consistency.core_anchors import (
    get_ikhtilaf_entry,
    is_core_principle_violation,
)
from consistency.models import (
    Contradiction,
    ContradictionCategory,
    ContradictionSeverity,
    FactualClaim,
    RulingType,
)


def _token_similarity(text1: str, text2: str) -> float:
    """Compute Jaccard token similarity between two normalized strings."""
    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())
    if not tokens1 or not tokens2:
        return 0.0
    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    return intersection / union if union > 0 else 0.0


def _are_rulings_incompatible(ruling1: RulingType | None, ruling2: RulingType | None, pol1: bool, pol2: bool) -> bool:
    """Determine whether two rulings logically oppose each other."""
    if ruling1 is None or ruling2 is None:
        # Check pure polarity clash
        return pol1 != pol2

    # Prohibitions vs Permissibility
    prohibitions = {RulingType.HARAM, RulingType.INVALID, RulingType.NULLIFIED}
    permissibilities = {RulingType.HALAL, RulingType.VALID, RulingType.NOT_NULLIFIED, RulingType.MUBAH}
    obligations = {RulingType.FARD, RulingType.WAJIB}

    if ruling1 in prohibitions and ruling2 in permissibilities:
        return True
    if ruling2 in prohibitions and ruling1 in permissibilities:
        return True
    if ruling1 in prohibitions and ruling2 in obligations:
        return True
    if ruling2 in prohibitions and ruling1 in obligations:
        return True

    # Polarity difference with same topic
    if pol1 != pol2 and ruling1 != RulingType.CONDITIONAL and ruling2 != RulingType.CONDITIONAL:
        return True

    return False


def detect_contradictions(
    candidate_claims: list[FactualClaim],
    historical_claims: list[FactualClaim],
) -> list[Contradiction]:
    """Detect inconsistencies between candidate claims and past history / anchors."""
    contradictions: list[Contradiction] = []

    for candidate in candidate_claims:
        # 1. Check against immutable Core Anchors (Aqeedah / Ijma)
        is_violation, reason = is_core_principle_violation(candidate)
        if is_violation and reason:
            contradictions.append(
                Contradiction(
                    historical_claim=None,
                    candidate_claim=candidate,
                    severity=ContradictionSeverity.CRITICAL,
                    category=ContradictionCategory.CORE_POSITION_DIVERGENCE,
                    description=reason,
                    is_legitimate_variation=False,
                    confidence=1.0,
                )
            )
            continue

        # 2. Check against Historical Claims
        for hist in historical_claims:
            # Skip comparing identical claim instance
            if hist.claim_id == candidate.claim_id:
                continue

            # Entity or topic match
            same_entity = candidate.entity is not None and candidate.entity == hist.entity
            sim = _token_similarity(candidate.normalized_text or candidate.text, hist.normalized_text or hist.text)
            is_related = same_entity or (candidate.topic == hist.topic and sim >= 0.35) or sim >= 0.60

            if not is_related:
                continue

            # --- Check A: Ruling Contradiction ---
            if _are_rulings_incompatible(candidate.ruling, hist.ruling, candidate.polarity, hist.polarity):
                # Check if this is a legitimate madhhab or conditional variation
                ikhtilaf_info = get_ikhtilaf_entry(candidate.entity) or get_ikhtilaf_entry(hist.entity)
                is_legit = False
                legit_reason: str | None = None
                reconciliation: str | None = None

                def _extract_school_key(madhhab: str | None, attribution: str | None) -> str | None:
                    if madhhab:
                        return madhhab.lower()
                    if attribution:
                        attr_l = attribution.lower()
                        for s_key in ("shafii", "shafi'i", "hanafi", "maliki", "hanbali"):
                            if s_key in attr_l:
                                return "shafii" if "shafi" in s_key else s_key
                        if "abu hanifa" in attr_l:
                            return "hanafi"
                        if "malik" in attr_l:
                            return "maliki"
                        if "ahmad" in attr_l or "hanbal" in attr_l:
                            return "hanbali"
                    return None

                cand_school = _extract_school_key(candidate.madhhab, candidate.attribution)
                hist_school = _extract_school_key(hist.madhhab, hist.attribution)

                # Check if candidate claims a ruling contrary to its own stated school's known position
                is_false_school_ruling = False
                if ikhtilaf_info and cand_school and cand_school in ikhtilaf_info.get("positions", {}):
                    known_school_pos = ikhtilaf_info["positions"][cand_school]
                    known_ruling = known_school_pos.get("ruling")
                    known_pol = known_ruling in (
                        RulingType.HALAL,
                        RulingType.VALID,
                        RulingType.NOT_NULLIFIED,
                        RulingType.MUSTAHABB,
                        RulingType.FARD,
                        RulingType.WAJIB,
                    )
                    if _are_rulings_incompatible(candidate.ruling, known_ruling, candidate.polarity, known_pol):
                        is_false_school_ruling = True

                same_school = cand_school is not None and hist_school is not None and cand_school == hist_school

                # Different madhhabs specified and not attributing false position to a school
                if cand_school and hist_school and cand_school != hist_school and not is_false_school_ruling:
                    is_legit = True
                    legit_reason = "madhhab_difference"
                # Different conditions specified (e.g. traveler vs resident, with desire vs without)
                elif candidate.condition and hist.condition and candidate.condition != hist.condition:
                    is_legit = True
                    legit_reason = "conditional_context"
                # Subject is a known classical Ikhtilaf topic (when no same-school conflict and not false school ruling)
                elif ikhtilaf_info is not None and not same_school and not is_false_school_ruling:
                    is_legit = True
                    legit_reason = "classical_ikhtilaf_topic"
                    reconciliation = ikhtilaf_info.get("reconciliation_template")

                severity = ContradictionSeverity.LOW if is_legit else ContradictionSeverity.HIGH
                desc = (
                    f"Ruling clash on '{candidate.entity or candidate.topic}': "
                    f"Candidate states '{candidate.ruling or ('positive' if candidate.polarity else 'negative')}' "
                    f"while previous session stated '{hist.ruling or ('positive' if hist.polarity else 'negative')}'."
                )

                contradictions.append(
                    Contradiction(
                        historical_claim=hist,
                        candidate_claim=candidate,
                        severity=severity,
                        category=ContradictionCategory.RULING_CONTRADICTION,
                        description=desc,
                        is_legitimate_variation=is_legit,
                        legitimate_reason=legit_reason,
                        reconciliation_text=reconciliation,
                        confidence=0.95,
                    )
                )

            # --- Check B: Attribution Mismatch ---
            elif candidate.attribution and hist.attribution and candidate.attribution != hist.attribution:
                if same_entity or sim >= 0.50:
                    desc = (
                        f"Scholarly attribution mismatch on '{candidate.entity or candidate.topic}': "
                        f"Currently attributed to '{candidate.attribution}', previously to '{hist.attribution}'."
                    )
                    contradictions.append(
                        Contradiction(
                            historical_claim=hist,
                            candidate_claim=candidate,
                            severity=ContradictionSeverity.MEDIUM,
                            category=ContradictionCategory.ATTRIBUTION_MISMATCH,
                            description=desc,
                            is_legitimate_variation=False,
                            legitimate_reason="attribution_drift",
                            confidence=0.90,
                        )
                    )

            # --- Check C: Numerical Discrepancy ---
            elif candidate.claim_type == hist.claim_type and "zakat" in candidate.topic:
                cand_nums = set(re.findall(r"\b\d+(?:\.\d+)?%?", candidate.text))
                hist_nums = set(re.findall(r"\b\d+(?:\.\d+)?%?", hist.text))
                if cand_nums and hist_nums and cand_nums != hist_nums:
                    desc = (
                        f"Numerical discrepancy in {candidate.topic}: "
                        f"Candidate mentions {sorted(cand_nums)}, previously stated {sorted(hist_nums)}."
                    )
                    contradictions.append(
                        Contradiction(
                            historical_claim=hist,
                            candidate_claim=candidate,
                            severity=ContradictionSeverity.HIGH,
                            category=ContradictionCategory.NUMERICAL_DISCREPANCY,
                            description=desc,
                            is_legitimate_variation=False,
                            confidence=0.92,
                        )
                    )

    return contradictions
