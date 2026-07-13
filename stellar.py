"""Stellar integration for the Deen Bridge AI service.

Deen Bridge runs on the Stellar network: users hold USDC in their own
wallets and purchase courses and books with it. This module gives the AI
service read-only Stellar awareness, starting with zakat calculation on a
wallet's on-chain USDC balance.

Strictly read-only: only public keys ever reach this service. Secret keys
are never accepted, stored, or logged.
"""

import logging
import os
from decimal import Decimal
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from stellar_sdk import Server
from stellar_sdk.exceptions import NotFoundError
from stellar_sdk.strkey import StrKey

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
# 85g of gold (or 595g of silver) and changes with market prices, so it
# is configurable; consult a scholar for rulings.
ZAKAT_RATE = Decimal("0.025")
DEFAULT_NISAB_USD = Decimal(os.getenv("ZAKAT_NISAB_USD", "6000"))

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
        "features": ["zakat"],
    }


@router.post("/zakat", response_model=ZakatResponse)
async def calculate_zakat(request: ZakatRequest):
    """Calculate zakat due on a wallet's on-chain USDC balance."""
    public_key = request.public_key.strip()
    if not StrKey.is_valid_ed25519_public_key(public_key):
        raise HTTPException(
            status_code=400,
            detail="Invalid Stellar public key. Expected a 56-character key starting with G.",
        )

    logger.info("Zakat lookup for %s on %s", public_key[:8], STELLAR_NETWORK)
    balance = fetch_usdc_balance(public_key)
    nisab = Decimal(str(request.nisab_usd)) if request.nisab_usd else DEFAULT_NISAB_USD

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
        )

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
    )
