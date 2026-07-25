"""Stellar integration for the Deen Bridge AI service.

Deen Bridge runs on the Stellar network: users hold USDC in their own
wallets and purchase courses and books with it. This module gives the AI
service read-only Stellar awareness, starting with zakat calculation on a
wallet's on-chain USDC balance.

Strictly read-only: only public keys ever reach this service. Secret keys
are never accepted, stored, or logged.
"""

import asyncio
import logging
import os
import re
from decimal import Decimal
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from stellar_sdk import Server
from stellar_sdk.exceptions import NotFoundError
from stellar_sdk.strkey import StrKey

from nisab import NisabQuote, get_nisab, override_quote

logger = logging.getLogger(__name__)

router = APIRouter(tags=["stellar"])

# Network configuration — must match the rest of the platform
# (dnb-backend uses the same issuers in src/services/stellar/stellarService.js)
STELLAR_NETWORK = os.getenv("STELLAR_NETWORK", "testnet")

HORIZON_URLS = {
    "testnet": "https://horizon-testnet.stellar.org",
    "public": "https://horizon.stellar.org",
}

USDC_ISSUERS = {
    # Circle's official USDC issuer on mainnet
    "public": "GA5ZSEJYB37JRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN",
    # Test USDC issuer used across the Deen Bridge platform on testnet
    "testnet": "GBBD47IF6LWK7P7MDEVSCWR7DPUWV3NY3DTQEVFL4NAT4AQH3ZLLFLA5",
}

# Zakat is 2.5% of zakatable wealth held for a lunar year, due once the
# total meets the nisab threshold. The nisab derives from the value of
# 85g of gold (or 595g of silver) and changes with market prices, so it is
# fetched live from the gold price (see nisab.py) with the configured
# ZAKAT_NISAB_USD as a fallback; consult a scholar for rulings.
ZAKAT_RATE = Decimal("0.025")

# USDC has 7 decimal places on Stellar, so zakat is quantized to the smallest
# unit the asset can actually represent.
USDC_PRECISION = Decimal("0.0000001")

DISCLAIMER = (
    "This is an automated estimate based on your on-chain USDC balance only. "
    "Zakat rulings depend on your full wealth, debts, and the hawl (lunar year). "
    "Please consult a qualified scholar for a definitive ruling."
)


def horizon_url() -> str:
    return HORIZON_URLS.get(STELLAR_NETWORK, HORIZON_URLS["testnet"])


def usdc_issuer() -> str:
    return USDC_ISSUERS.get(STELLAR_NETWORK, USDC_ISSUERS["testnet"])


class ZakatRequest(BaseModel):
    public_key: str = Field(..., description="Stellar account public key (G...)")
    nisab_usd: Optional[float] = Field(
        None, gt=0, description="Override the nisab threshold in USD"
    )


class ZakatResponse(BaseModel):
    network: str
    public_key: str
    has_usdc_trustline: bool
    usdc_balance: str
    nisab_usd: str
    zakat_rate: str
    zakat_due: str
    message: str
    disclaimer: str
    nisab: Optional[NisabQuote] = Field(
        None,
        description=(
            "Where the nisab threshold came from — a live gold price, the "
            "configured default, or a caller override."
        ),
    )


def compute_zakat(balance: Decimal, nisab: Decimal) -> Decimal:
    """Zakat due on *balance*, or zero when it does not reach *nisab*.

    Zakat is all-or-nothing against the threshold: wealth below the nisab owes
    nothing at all, and wealth at or above it owes 2.5% of the *whole* amount,
    not of the excess.
    """
    if balance < nisab:
        return Decimal("0")
    return (balance * ZAKAT_RATE).quantize(USDC_PRECISION)


def validate_public_key(public_key: str) -> str:
    """Return the trimmed key, or raise HTTPException(400).

    Read-only by construction: a secret key (S...) fails this check like any
    other malformed input, so one can never reach Horizon from here.
    """
    cleaned = (public_key or "").strip()
    if not StrKey.is_valid_ed25519_public_key(cleaned):
        raise HTTPException(
            status_code=400,
            detail="Invalid Stellar public key. Expected a 56-character key starting with G.",
        )
    return cleaned


def build_zakat_response(
    public_key: str, balance: Optional[Decimal], nisab_quote: NisabQuote
) -> ZakatResponse:
    """Assemble the response from an already-fetched balance and nisab.

    Pure: no network, so the wording and the arithmetic are testable offline.
    """
    nisab = nisab_quote.amount
    basis = (
        f"the nisab threshold of {nisab} USD "
        f"({'live gold price via ' + nisab_quote.source if nisab_quote.live else nisab_quote.source})"
    )

    if balance is None:
        return ZakatResponse(
            network=STELLAR_NETWORK,
            public_key=public_key,
            has_usdc_trustline=False,
            usdc_balance="0",
            nisab_usd=str(nisab),
            zakat_rate=str(ZAKAT_RATE),
            zakat_due="0",
            message=(
                "This account has no USDC trustline, so it holds no USDC. "
                "Add a USDC trustline in your wallet to hold USDC on Stellar."
            ),
            disclaimer=DISCLAIMER,
            nisab=nisab_quote,
        )

    zakat_due = compute_zakat(balance, nisab)
    if balance >= nisab:
        message = (
            f"Your USDC balance of {balance} meets {basis}. "
            f"If held for a full lunar year, the zakat due is {zakat_due} USDC (2.5%)."
        )
    else:
        message = (
            f"Your USDC balance of {balance} is below {basis}, "
            "so no zakat is due on this balance alone."
        )

    return ZakatResponse(
        network=STELLAR_NETWORK,
        public_key=public_key,
        has_usdc_trustline=True,
        usdc_balance=str(balance),
        nisab_usd=str(nisab),
        zakat_rate=str(ZAKAT_RATE),
        zakat_due=str(zakat_due),
        message=message,
        disclaimer=DISCLAIMER,
        nisab=nisab_quote,
    )


def fetch_usdc_balance(public_key: str) -> Optional[Decimal]:
    """Return the account's USDC balance, or None if it has no USDC trustline.

    Raises HTTPException(404) if the account does not exist on this network.
    """
    server = Server(horizon_url())
    try:
        account = server.accounts().account_id(public_key).call()
    except NotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Account not found on the Stellar {STELLAR_NETWORK} network.",
        )
    for balance in account.get("balances", []):
        if (
            balance.get("asset_code") == "USDC"
            and balance.get("asset_issuer") == usdc_issuer()
        ):
            return Decimal(balance["balance"])
    return None


@router.get("/stellar/info")
async def stellar_info():
    """Public configuration of this service's Stellar integration."""
    return {
        "network": STELLAR_NETWORK,
        "horizon": horizon_url(),
        "usdc_issuer": usdc_issuer(),
        "features": ["zakat", "chat-zakat", "live-nisab"],
    }


# ---------------------------------------------------------------------------
# Chat integration
# ---------------------------------------------------------------------------

ZAKAT_KEYWORDS = (
    "zakat", "zakah", "zakaat", "nisab", "nisaab", "almsgiving", "alms",
)

# Stellar public keys are 56 characters of base32 starting with G. Matching
# the shape first keeps StrKey validation (and any Horizon call) off every
# ordinary message.
PUBLIC_KEY_PATTERN = re.compile(r"\bG[A-Z2-7]{55}\b")

# A secret key has the same shape but starts with S. It is matched *only* so
# it can be recognized and refused — it is never validated, stored, logged, or
# sent anywhere.
SECRET_KEY_PATTERN = re.compile(r"\bS[A-Z2-7]{55}\b")

SECRET_KEY_WARNING = (
    "\n\nIMPORTANT: the user's message appears to contain a Stellar SECRET key. "
    "Do not repeat it back. Warn them plainly that a secret key must never be "
    "shared with anyone or pasted into any website or chat, that this service "
    "only ever needs their public key (starting with G), and that they should "
    "treat the key as compromised and move their funds to a new wallet."
)


def is_zakat_question(text: str) -> bool:
    lowered = (text or "").casefold()
    return any(keyword in lowered for keyword in ZAKAT_KEYWORDS)


def extract_public_key(*texts: Optional[str]) -> Optional[str]:
    """First valid Stellar public key found across *texts*, if any."""
    for text in texts:
        if not text:
            continue
        for candidate in PUBLIC_KEY_PATTERN.findall(text):
            if StrKey.is_valid_ed25519_public_key(candidate):
                return candidate
    return None


def contains_secret_key(*texts: Optional[str]) -> bool:
    """True if any text looks like it contains a Stellar secret key."""
    return any(SECRET_KEY_PATTERN.search(text or "") for text in texts)


SECRET_KEY_PLACEHOLDER = "[REDACTED STELLAR SECRET KEY]"


def redact_secret_keys(text: Optional[str]) -> Optional[str]:
    """Replace any secret-key-shaped string with a placeholder.

    A user who pastes their secret key into the chat should not have it
    forwarded to the model provider or written into stored conversation
    history. This module promises secret keys are never stored or logged; that
    promise has to survive a user volunteering one.
    """
    if not text:
        return text
    return SECRET_KEY_PATTERN.sub(SECRET_KEY_PLACEHOLDER, text)


class ZakatInfo(BaseModel):
    """What the zakat integration contributed to a chat answer."""

    calculated: bool = Field(
        ..., description="True when an on-chain balance was actually read"
    )
    public_key: Optional[str] = Field(
        None, description="The wallet the figures describe"
    )
    usdc_balance: Optional[str] = None
    nisab_usd: Optional[str] = None
    zakat_due: Optional[str] = None
    nisab_source: Optional[str] = Field(
        None, description="Where the nisab threshold came from"
    )
    secret_key_detected: bool = Field(
        False,
        description=(
            "The message appeared to contain a secret key; the answer warns "
            "the user instead of using it."
        ),
    )


class ZakatContext(BaseModel):
    """Retrieved zakat figures for a chat turn, plus the prompt block."""

    info: ZakatInfo
    prompt_block: str


def build_zakat_prompt_block(response: ZakatResponse) -> str:
    """Render a computed zakat result as grounding for the model."""
    lines = [
        "",
        "ZAKAT CALCULATION (real on-chain data for this user):",
        f"- Wallet: {response.public_key}",
        f"- Network: {response.network}",
        f"- USDC balance: {response.usdc_balance}",
        f"- Nisab threshold: {response.nisab_usd} USD",
    ]
    if response.nisab:
        basis = "live gold price" if response.nisab.live else "fallback"
        lines.append(f"- Nisab basis: 85g gold, {basis}, via {response.nisab.source}")
        if response.nisab.gold_price_usd_per_ounce:
            lines.append(
                f"- Gold price used: {response.nisab.gold_price_usd_per_ounce} USD/troy ounce"
            )
    lines += [
        f"- Meets nisab: {'yes' if response.has_usdc_trustline and Decimal(response.usdc_balance) >= Decimal(response.nisab_usd) else 'no'}",
        f"- Zakat due (2.5%): {response.zakat_due} USDC",
        "",
        "Use these figures — they are this user's real balance, not an example. "
        "State the numbers plainly, explain that zakat is 2.5% of zakatable "
        "wealth held for a full lunar year (hawl) once the nisab is met, and "
        "note that this covers only their on-chain USDC, not their other "
        "wealth or debts. Close with this disclaimer verbatim: "
        f'"{DISCLAIMER}"',
    ]
    return "\n".join(lines)


NO_KEY_NOTE = (
    "\n\nZAKAT QUESTION WITHOUT A WALLET: the user asked about zakat but gave "
    "no Stellar public key, so no balance could be read. Explain how zakat is "
    "calculated (2.5% of zakatable wealth held for a lunar year, once it meets "
    "the nisab — the value of 85g of gold), and invite them to share their "
    "Stellar public key (starting with G) if they want it calculated from "
    "their on-chain USDC balance. Never ask for a secret key."
)


async def build_chat_zakat_context(
    prompt: str, context: Optional[str] = None
) -> Optional[ZakatContext]:
    """Compute zakat for a chat turn, or None if this isn't a zakat question.

    Detection is offline (keywords plus a key-shaped match), so an ordinary
    message never touches Horizon or the gold-price API.
    """
    if not is_zakat_question(prompt):
        return None

    if contains_secret_key(prompt, context):
        logger.warning("Secret-key-shaped string in a chat message; refusing to use it")
        return ZakatContext(
            info=ZakatInfo(calculated=False, secret_key_detected=True),
            prompt_block=SECRET_KEY_WARNING,
        )

    public_key = extract_public_key(prompt, context)
    if public_key is None:
        return ZakatContext(
            info=ZakatInfo(calculated=False),
            prompt_block=NO_KEY_NOTE,
        )

    try:
        result = await zakat_for_account(public_key)
    except HTTPException as exc:
        # An unfunded or unknown account is a normal answer, not an error.
        logger.info("Zakat lookup for chat failed: %s", exc.detail)
        return ZakatContext(
            info=ZakatInfo(calculated=False, public_key=public_key),
            prompt_block=(
                f"\n\nZAKAT LOOKUP FAILED: {exc.detail} Tell the user plainly "
                "that their wallet could not be read on this network, and "
                "explain how zakat would be calculated if it could."
            ),
        )

    return ZakatContext(
        info=ZakatInfo(
            calculated=True,
            public_key=public_key,
            usdc_balance=result.usdc_balance,
            nisab_usd=result.nisab_usd,
            zakat_due=result.zakat_due,
            nisab_source=result.nisab.source if result.nisab else None,
        ),
        prompt_block=build_zakat_prompt_block(result),
    )


async def resolve_nisab(nisab_usd_override: Optional[float]) -> NisabQuote:
    """Caller's override if given, otherwise the live gold-derived nisab."""
    if nisab_usd_override:
        return override_quote(Decimal(str(nisab_usd_override)))
    return await get_nisab()


async def zakat_for_account(
    public_key: str, nisab_usd_override: Optional[float] = None
) -> ZakatResponse:
    """Full zakat calculation for a validated public key.

    Shared by ``POST /zakat`` and the chat integration so both report the same
    figures, the same nisab basis, and the same disclaimer.
    """
    logger.info("Zakat lookup for %s on %s", public_key[:8], STELLAR_NETWORK)
    # stellar_sdk's Server is synchronous (requests-based), so it runs off the
    # event loop rather than blocking every other in-flight request.
    balance, nisab_quote = await asyncio.gather(
        asyncio.to_thread(fetch_usdc_balance, public_key),
        resolve_nisab(nisab_usd_override),
    )
    return build_zakat_response(public_key, balance, nisab_quote)


@router.post("/zakat", response_model=ZakatResponse)
async def calculate_zakat(request: ZakatRequest):
    """Calculate zakat due on a wallet's on-chain USDC balance."""
    public_key = validate_public_key(request.public_key)
    return await zakat_for_account(public_key, request.nisab_usd)
