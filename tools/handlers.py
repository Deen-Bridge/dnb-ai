"""Tool handler implementations for the Gemini function-calling framework.

Every handler is a sync callable that returns a dict (→ function_response).
All handlers are read-only against external systems.
"""

import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, Field
from stellar_sdk import Server
from stellar_sdk.exceptions import NotFoundError
from stellar_sdk.strkey import StrKey

from .registry import Tool, ToolRegistry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared Stellar config (mirrors stellar.py)
# ---------------------------------------------------------------------------

STELLAR_NETWORK = os.getenv("STELLAR_NETWORK", "testnet")

HORIZON_URLS = {
    "testnet": "https://horizon-testnet.stellar.org",
    "public": "https://horizon.stellar.org",
}

USDC_ISSUERS = {
    "public": "GA5ZSEJYB37JRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN",
    "testnet": "GBBD47IF6LWK7P7MDEVSCWR7DPUWV3NY3DTQEVFL4NAT4AQH3ZLLFLA5",
}

ZAKAT_RATE = Decimal("0.025")
DEFAULT_NISAB_USD = Decimal(os.getenv("ZAKAT_NISAB_USD", "6000"))

DISCLAIMER = (
    "This is an automated estimate based on your on-chain USDC balance only. "
    "Zakat rulings depend on your full wealth, debts, and the hawl (lunar year). "
    "Please consult a qualified scholar for a definitive ruling."
)


def _horizon_url() -> str:
    return HORIZON_URLS.get(STELLAR_NETWORK, HORIZON_URLS["testnet"])


def _usdc_issuer() -> str:
    return USDC_ISSUERS.get(STELLAR_NETWORK, USDC_ISSUERS["testnet"])


def _fetch_usdc_balance(public_key: str) -> Optional[Decimal]:
    """Return the account's USDC balance, or None if it has no USDC trustline."""
    server = Server(_horizon_url())
    try:
        account = server.accounts().account_id(public_key).call()
    except NotFoundError:
        return None
    for balance in account.get("balances", []):
        if (
            balance.get("asset_code") == "USDC"
            and balance.get("asset_issuer") == _usdc_issuer()
        ):
            return Decimal(balance["balance"])
    return None


# ---------------------------------------------------------------------------
# Tool: calculate_zakat
# ---------------------------------------------------------------------------


class CalculateZakatArgs(BaseModel):
    public_key: str = Field(..., description="Stellar account public key (G...)")
    nisab_usd: Optional[float] = Field(None, description="Override nisab threshold in USD")


def _calculate_zakat(public_key: str, nisab_usd: Optional[float] = None) -> dict:
    if not StrKey.is_valid_ed25519_public_key(public_key):
        return {"error": "Invalid Stellar public key. Expected a 56-character key starting with G."}

    logger.info("Zakat lookup for %s on %s", public_key[:8], STELLAR_NETWORK)
    balance = _fetch_usdc_balance(public_key)
    nisab = Decimal(str(nisab_usd)) if nisab_usd else DEFAULT_NISAB_USD

    if balance is None:
        return {
            "network": STELLAR_NETWORK,
            "public_key": public_key,
            "has_usdc_trustline": False,
            "usdc_balance": "0",
            "nisab_usd": str(nisab),
            "zakat_rate": str(ZAKAT_RATE),
            "zakat_due": "0",
            "message": "This account has no USDC trustline.",
            "disclaimer": DISCLAIMER,
        }

    meets_nisab = balance >= nisab
    zakat_due = (balance * ZAKAT_RATE).quantize(Decimal("0.0000001")) if meets_nisab else Decimal("0")

    if meets_nisab:
        message = (
            f"Your USDC balance of {balance} meets the nisab threshold of {nisab} USD. "
            f"If held for a full lunar year, the zakat due is {zakat_due} USDC (2.5%)."
        )
    else:
        message = (
            f"Your USDC balance of {balance} is below the nisab threshold of {nisab} USD, "
            "so no zakat is due on this balance alone."
        )

    return {
        "network": STELLAR_NETWORK,
        "public_key": public_key,
        "has_usdc_trustline": True,
        "usdc_balance": str(balance),
        "nisab_usd": str(nisab),
        "zakat_rate": str(ZAKAT_RATE),
        "zakat_due": str(zakat_due),
        "message": message,
        "disclaimer": DISCLAIMER,
    }


CALCULATE_ZAKAT_TOOL = Tool(
    name="calculate_zakat",
    description=(
        "Calculate the zakat (obligatory charity) due on a Stellar wallet's "
        "on-chain USDC balance. Returns the balance, nisab threshold, "
        "zakat rate (2.5%), and the calculated amount due. Includes a "
        "scholar-consultation disclaimer."
    ),
    args_schema=CalculateZakatArgs,
    handler=_calculate_zakat,
    timeout_seconds=15,
)

# ---------------------------------------------------------------------------
# Tool: get_stellar_info
# ---------------------------------------------------------------------------


class GetStellarInfoArgs(BaseModel):
    pass


def _get_stellar_info() -> dict:
    return {
        "network": STELLAR_NETWORK,
        "horizon": _horizon_url(),
        "usdc_issuer": _usdc_issuer(),
        "features": ["zakat"],
    }


GET_STELLAR_INFO_TOOL = Tool(
    name="get_stellar_info",
    description="Get configuration information about the Stellar network integration used by Deen Bridge.",
    args_schema=GetStellarInfoArgs,
    handler=_get_stellar_info,
    timeout_seconds=5,
)

# ---------------------------------------------------------------------------
# Tool: search_courses
# ---------------------------------------------------------------------------


class SearchCoursesArgs(BaseModel):
    query: str = Field(..., description="Search query for course titles or descriptions")


BACKEND_API_URL = os.getenv("BACKEND_API_URL", "https://dnb-backend-api.onrender.com")


def _search_courses(query: str) -> dict:
    url = f"{BACKEND_API_URL.rstrip('/')}/api/courses/search?q={urllib.parse.quote(query)}"
    logger.info("Course search query=%s url=%s", query[:80], url)
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode())
        return {"results": data, "count": len(data) if isinstance(data, list) else 1}
    except urllib.error.HTTPError as e:
        return {"error": f"Backend returned HTTP {e.code}: {e.reason}"}
    except urllib.error.URLError as e:
        return {"error": f"Cannot reach course backend: {e.reason}"}
    except Exception as e:
        return {"error": f"Course search failed: {e}"}


SEARCH_COURSES_TOOL = Tool(
    name="search_courses",
    description=(
        "Search for Islamic courses available on the Deen Bridge platform. "
        "Returns matching course titles, descriptions, and enrollment details."
    ),
    args_schema=SearchCoursesArgs,
    handler=_search_courses,
    timeout_seconds=10,
)

# ---------------------------------------------------------------------------
# Default registry
# ---------------------------------------------------------------------------

DEFAULT_TOOLS = [
    CALCULATE_ZAKAT_TOOL,
    GET_STELLAR_INFO_TOOL,
    SEARCH_COURSES_TOOL,
]


def get_registry() -> ToolRegistry:
    reg = ToolRegistry()
    for tool in DEFAULT_TOOLS:
        reg.register(tool)
    return reg
