"""Live nisab threshold, derived from the current gold price.

Why this exists
---------------
Zakat is due once a person's zakatable wealth reaches the *nisab* — the value
of 85 grams of gold (or 595 grams of silver). That is a **price**, not a
constant: it moves with the gold market, so a hardcoded figure drifts out of
date and quietly gives people the wrong answer in both directions. This module
derives the nisab from a live gold price, caches it, and falls back to the
configured default when no source can be reached.

The gold standard is used rather than silver because it is the more common
contemporary basis for cash-wealth nisab; the two differ substantially, and
which one applies is a matter scholars differ on. That choice is reported in
every quote (``basis``) rather than left implicit.

Price sources
-------------
Both are public and need no API key:

1. **gold-api.com** — spot XAU in USD per troy ounce.
2. **CoinGecko / PAX Gold** — the PAXG token is redeemable one-for-one for a
   troy ounce of allocated London Good Delivery gold, so its USD price tracks
   spot closely. Used only when the primary source is unreachable.

If both fail, the quote falls back to ``ZAKAT_NISAB_USD`` and is marked
``live: false`` so a caller can tell the difference between a live figure and a
stale default.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel, Field

from semantic_cache import get_keyed_cache

logger = logging.getLogger(__name__)

# 85 grams of gold, the classical nisab, in troy ounces.
NISAB_GOLD_GRAMS = Decimal("85")
GRAMS_PER_TROY_OUNCE = Decimal("31.1034768")

DEFAULT_NISAB_USD = Decimal(os.getenv("ZAKAT_NISAB_USD", "6000"))

GOLD_PRICE_TIMEOUT = float(os.getenv("GOLD_PRICE_TIMEOUT", "8"))
# Gold moves continuously, but a nisab that is hours old is still a faithful
# threshold — and refetching per request would be both slow and rude to a free
# public API.
NISAB_CACHE_TTL_SECONDS = int(os.getenv("NISAB_CACHE_TTL_SECONDS", "21600"))  # 6h

CACHE_NAMESPACE = "nisab"
CACHE_KEY = "gold-nisab-usd"

# A price outside this range is a source error (wrong field, wrong unit,
# wrong currency), not a market move. Refusing it is safer than computing a
# nisab from a number that cannot be a gold price.
MIN_PLAUSIBLE_OUNCE_USD = Decimal("100")
MAX_PLAUSIBLE_OUNCE_USD = Decimal("100000")


class GoldPriceSource(BaseModel):
    """One place to ask for the spot gold price."""

    name: str
    url: str
    # Dotted path to the price within the JSON response.
    path: List[str]


PRIMARY_SOURCE = GoldPriceSource(
    name="gold-api.com",
    url="https://api.gold-api.com/price/XAU",
    path=["price"],
)

FALLBACK_SOURCE = GoldPriceSource(
    name="CoinGecko (PAX Gold)",
    url="https://api.coingecko.com/api/v3/simple/price?ids=pax-gold&vs_currencies=usd",
    path=["pax-gold", "usd"],
)

GOLD_PRICE_SOURCES = (PRIMARY_SOURCE, FALLBACK_SOURCE)


class NisabQuote(BaseModel):
    """The nisab in force for a calculation, and where the number came from."""

    nisab_usd: str = Field(..., description="Nisab threshold in USD")
    live: bool = Field(
        ...,
        description=(
            "True when derived from a live gold price; false when the "
            "configured ZAKAT_NISAB_USD default was used because no price "
            "source could be reached."
        ),
    )
    source: str = Field(..., description="Where the figure came from")
    basis: str = Field(
        "85g gold",
        description="Which classical nisab basis was used",
    )
    gold_price_usd_per_ounce: Optional[str] = Field(
        None, description="Spot gold price used, per troy ounce"
    )
    gold_price_usd_per_gram: Optional[str] = Field(
        None, description="Spot gold price used, per gram"
    )
    as_of: Optional[str] = Field(
        None, description="UTC timestamp of the price fetch (ISO 8601)"
    )
    note: Optional[str] = Field(
        None, description="Explanation when a fallback was used"
    )

    @property
    def amount(self) -> Decimal:
        return Decimal(self.nisab_usd)


def _dig(payload: Any, path: List[str]) -> Any:
    for key in path:
        if not isinstance(payload, dict) or key not in payload:
            return None
        payload = payload[key]
    return payload


def parse_price(payload: Any, source: GoldPriceSource) -> Optional[Decimal]:
    """Pull the price out of a source's payload, or None if it is unusable.

    Rejects prices outside a plausible range: a source that changes its units
    or field meaning should degrade to the next source, not silently produce a
    nisab off by an order of magnitude.
    """
    raw = _dig(payload, source.path)
    if raw is None:
        logger.warning("%s response had no %s field", source.name, ".".join(source.path))
        return None
    try:
        price = Decimal(str(raw))
    except (InvalidOperation, ValueError):
        logger.warning("%s returned a non-numeric price: %r", source.name, raw)
        return None
    if not MIN_PLAUSIBLE_OUNCE_USD <= price <= MAX_PLAUSIBLE_OUNCE_USD:
        logger.warning(
            "%s returned an implausible gold price (%s USD/oz); ignoring it",
            source.name,
            price,
        )
        return None
    return price


def nisab_from_ounce_price(price_per_ounce: Decimal) -> Decimal:
    """Convert a spot price per troy ounce into the 85g-gold nisab, in USD."""
    per_gram = price_per_ounce / GRAMS_PER_TROY_OUNCE
    return (per_gram * NISAB_GOLD_GRAMS).quantize(Decimal("0.01"))


async def fetch_gold_price(
    client: Optional[httpx.AsyncClient] = None,
) -> Optional[Dict[str, Any]]:
    """Return ``{"price", "source"}`` from the first source that answers.

    Returns None when every source fails, which is the caller's cue to fall
    back to the configured default.
    """
    owns_client = client is None
    client = client or httpx.AsyncClient(timeout=GOLD_PRICE_TIMEOUT)
    try:
        for source in GOLD_PRICE_SOURCES:
            try:
                response = await client.get(source.url)
                response.raise_for_status()
                payload = response.json()
            except (httpx.HTTPError, ValueError) as exc:
                logger.warning("Gold price source %s failed: %s", source.name, exc)
                continue
            price = parse_price(payload, source)
            if price is not None:
                return {"price": price, "source": source.name}
        return None
    finally:
        if owns_client:
            await client.aclose()


def _fallback_quote(reason: str) -> NisabQuote:
    return NisabQuote(
        nisab_usd=str(DEFAULT_NISAB_USD),
        live=False,
        source="configured default (ZAKAT_NISAB_USD)",
        note=(
            f"{reason} The configured default was used instead, so this "
            "threshold may not reflect the current gold price."
        ),
    )


async def get_nisab(
    client: Optional[httpx.AsyncClient] = None,
    use_cache: bool = True,
) -> NisabQuote:
    """The nisab in force now: live from the gold price, or the default.

    Never raises — a price-source outage degrades the threshold, it does not
    fail the caller's zakat calculation.
    """
    cache = get_keyed_cache(CACHE_NAMESPACE)
    if use_cache:
        cached = cache.get(CACHE_KEY)
        if cached is not None:
            return NisabQuote(**cached)

    try:
        result = await fetch_gold_price(client)
    except Exception as exc:  # noqa: BLE001 - a price lookup must not break zakat
        logger.warning("Gold price lookup failed unexpectedly: %s", exc)
        result = None

    if result is None:
        return _fallback_quote("No live gold price source could be reached.")

    price_per_ounce: Decimal = result["price"]
    quote = NisabQuote(
        nisab_usd=str(nisab_from_ounce_price(price_per_ounce)),
        live=True,
        source=result["source"],
        gold_price_usd_per_ounce=str(price_per_ounce),
        gold_price_usd_per_gram=str(
            (price_per_ounce / GRAMS_PER_TROY_OUNCE).quantize(Decimal("0.01"))
        ),
        as_of=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    if use_cache:
        cache.put(CACHE_KEY, quote.model_dump(), ttl_seconds=NISAB_CACHE_TTL_SECONDS)
    return quote


def override_quote(nisab_usd: Decimal) -> NisabQuote:
    """A caller-supplied nisab, reported as such rather than as a live price."""
    return NisabQuote(
        nisab_usd=str(nisab_usd),
        live=False,
        source="caller-supplied override",
        note="This threshold was supplied in the request, not derived from a gold price.",
    )
