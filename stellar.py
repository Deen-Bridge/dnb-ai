"""Stellar integration for the Deen Bridge AI service.

Deen Bridge runs on the Stellar network: users hold USDC in their own
wallets and purchase courses and books with it. This module gives the AI
service read-only Stellar awareness, starting with zakat calculation on a
wallet's on-chain USDC balance.

Strictly read-only: only public keys ever reach this service. Secret keys
are never accepted, stored, or logged.

Pure computation lives in tools/handlers.py so the Gemini function-calling
tools and the REST endpoints share the same logic.
"""

import logging
from decimal import Decimal
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from stellar_sdk.strkey import StrKey

from tools.handlers import (
    DISCLAIMER,
    _calculate_zakat,
    _horizon_url,
    _usdc_issuer,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["stellar"])


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


def compute_zakat(public_key: str, nisab_usd: Optional[float] = None) -> ZakatResponse:
    """Pure computation shared by the REST endpoint and the tool handler.
    Returns a ZakatResponse; raises HTTPException on invalid key.
    """
    public_key = public_key.strip()
    if not StrKey.is_valid_ed25519_public_key(public_key):
        raise HTTPException(
            status_code=400,
            detail="Invalid Stellar public key. Expected a 56-character key starting with G.",
        )

    logger.info("Zakat lookup for %s", public_key[:8])
    result = _calculate_zakat(public_key, nisab_usd)

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    nisab_val = Decimal(result["nisab_usd"])
    zakat_due = Decimal(result["zakat_due"])

    return ZakatResponse(
        network=result["network"],
        public_key=result["public_key"],
        has_usdc_trustline=result["has_usdc_trustline"],
        usdc_balance=result["usdc_balance"],
        nisab_usd=str(nisab_val),
        zakat_rate=result["zakat_rate"],
        zakat_due=str(zakat_due),
        message=result["message"],
        disclaimer=result.get("disclaimer", DISCLAIMER),
    )


@router.get("/stellar/info")
async def stellar_info():
    """Public configuration of this service's Stellar integration."""
    return {
        "network": _horizon_url(),
        "horizon": _horizon_url(),
        "usdc_issuer": _usdc_issuer(),
        "features": ["zakat"],
    }


@router.post("/zakat", response_model=ZakatResponse)
async def calculate_zakat(request: ZakatRequest):
    """Calculate zakat due on a wallet's on-chain USDC balance."""
    return compute_zakat(request.public_key, request.nisab_usd)
