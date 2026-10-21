"""Distinguish Stellar protocol questions from Islamic jurisprudence requests (#584)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

STELLAR_KEYWORDS: frozenset[str] = frozenset(
    {
        "stellar",
        "soroban",
        "horizon",
        "trustline",
        "stroops",
        "strkey",
        "xlm",
        "usdc",
        "sep-10",
        "sep-24",
        "multisig",
        "ledger",
    }
)

ISLAMIC_KEYWORDS: frozenset[str] = frozenset(
    {
        "zakat",
        "sadaqah",
        "fiqh",
        "halal",
        "haram",
        "riba",
        "quran",
        "hadith",
        "madhhab",
        "fatwa",
        "waqf",
    }
)


@dataclass(frozen=True)
class DomainRoutingDecision:
    """Routing decision separating Stellar protocol docs from Islamic jurisprudence."""

    domain: str  # "stellar_protocol", "islamic_knowledge", "mixed_domain"
    required_corpora: tuple[str, ...]
    boundary_instruction: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def classify_query_domain(query: str) -> DomainRoutingDecision:
    """Classify a user query into protocol, Islamic knowledge, or mixed-domain."""
    lower = query.lower()
    has_stellar = any(kw in lower for kw in STELLAR_KEYWORDS)
    has_islamic = any(kw in lower for kw in ISLAMIC_KEYWORDS)

    if has_stellar and has_islamic:
        return DomainRoutingDecision(
            domain="mixed_domain",
            required_corpora=("official_stellar_docs", "authenticated_islamic_sources"),
            boundary_instruction=(
                "Keep technical Stellar protocol facts strictly grounded in official "
                "Stellar documentation and separate from Islamic jurisprudence references."
            ),
        )
    if has_stellar:
        return DomainRoutingDecision(
            domain="stellar_protocol",
            required_corpora=("official_stellar_docs",),
            boundary_instruction=(
                "Answer using official Stellar protocol documentation only; do not "
                "frame technical network rules as religious rulings."
            ),
        )
    return DomainRoutingDecision(
        domain="islamic_knowledge",
        required_corpora=("authenticated_islamic_sources",),
        boundary_instruction=(
            "Answer using authenticated Quran, Hadith, and scholarly sources; do not "
            "present religious guidance as Stellar protocol documentation."
        ),
    )
