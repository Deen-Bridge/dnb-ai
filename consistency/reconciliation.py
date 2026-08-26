"""Scholarly reconciliation and multi-perspective explanation generator.

Synthesizes differences across conversations into coherent, authentic Islamic
contextual explanations (Jam' bayn al-Aqwal), distinguishing valid diversity
of jurisprudence (Ikhtilaf) from errors.
"""

from __future__ import annotations

from consistency.core_anchors import get_ikhtilaf_entry
from consistency.models import Contradiction


def format_reconciliation_note(contradictions: list[Contradiction]) -> str:
    """Construct an educational reconciliation note for legitimate perspective variations."""
    notes: list[str] = []
    seen_entities: set[str] = set()

    for c in contradictions:
        if not c.is_legitimate_variation:
            continue

        entity = c.candidate_claim.entity or (c.historical_claim.entity if c.historical_claim else None)
        if entity and entity in seen_entities:
            continue
        if entity:
            seen_entities.add(entity)

        # 1. Use pre-built canonical template if available
        if c.reconciliation_text:
            notes.append(c.reconciliation_text)
            continue

        ikhtilaf = get_ikhtilaf_entry(entity)
        if ikhtilaf and "reconciliation_template" in ikhtilaf:
            notes.append(ikhtilaf["reconciliation_template"])
            continue

        # 2. Dynamic generation based on madhhab difference
        if c.legitimate_reason == "madhhab_difference" and c.historical_claim:
            cand_madhhab = (c.candidate_claim.madhhab or "one").capitalize()
            hist_madhhab = (c.historical_claim.madhhab or "another").capitalize()
            notes.append(
                f"Scholarly Perspective Note: There are valid differences of opinion (ikhtilaf) on this matter. "
                f"According to the {hist_madhhab} school, this ruling applies as previously discussed, "
                f"whereas in the {cand_madhhab} school, the ruling differs. Both views are respected traditions."
            )

        # 3. Dynamic generation based on conditional context
        elif c.legitimate_reason == "conditional_context" and c.historical_claim:
            cand_cond = c.candidate_claim.condition or "specific condition"
            hist_cond = c.historical_claim.condition or "general context"
            notes.append(
                f"Contextual Distinction: Rulings vary based on individual circumstances. "
                f"Under the condition of {cand_cond}, this ruling applies, whereas under {hist_cond}, a different standard is observed."
            )

        # 4. General classical ikhtilaf topic
        elif c.legitimate_reason == "classical_ikhtilaf_topic":
            notes.append(
                "Scholarly Perspective Note: This topic features recognized classical diversity of opinion (ikhtilaf) "
                "among the major schools of Islamic jurisprudence, with each school relying on established textual proofs."
            )

    if not notes:
        return ""

    joined = "\n\n".join(f"📌 **{n}**" for n in notes)
    return f"\n\n---\n### 📖 Scholarly Context & Reconciliation\n{joined}"


def reconcile_response_text(original_text: str, contradictions: list[Contradiction]) -> str:
    """Enhance response text with reconciliation notes for legitimate variations."""
    note = format_reconciliation_note(contradictions)
    if not note:
        return original_text
    return f"{original_text.rstrip()}{note}"
