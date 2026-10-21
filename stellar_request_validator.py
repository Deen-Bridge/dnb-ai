"""Validate network and asset parameters at the API boundary before provider access (#581)."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

_ASSET_CODE_RE = re.compile(r"^[A-Za-z0-9]{1,12}$")
_G_ADDRESS_RE = re.compile(r"^G[A-Z2-7]{55}$")
_TX_HASH_RE = re.compile(r"^[0-9a-fA-F]{64}$")


@dataclass(frozen=True)
class FieldValidationError:
    """Field-specific 422 validation error item."""

    field: str
    message: str


@dataclass(frozen=True)
class StellarBoundaryValidationResult:
    """API boundary validation result preventing invalid calls from hitting providers."""

    is_valid: bool
    status_code: int  # 200 or 422
    errors: tuple[FieldValidationError, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_stellar_tool_parameters(
    *,
    network: str,
    asset_code: str = "XLM",
    asset_issuer: str | None = None,
    tx_hash: str | None = None,
    amount: str | None = None,
) -> StellarBoundaryValidationResult:
    """Validate network, asset code, issuer, hash, and amount bounds locally."""
    errs: list[FieldValidationError] = []
    net = network.strip().lower()
    if net not in {"testnet", "public"}:
        errs.append(
            FieldValidationError(
                field="network",
                message="network must be either 'testnet' or 'public'.",
            )
        )

    code = asset_code.strip()
    if not _ASSET_CODE_RE.match(code):
        errs.append(
            FieldValidationError(
                field="asset_code",
                message="asset_code must be 1-12 alphanumeric characters.",
            )
        )
    elif code.upper() != "XLM":
        if not asset_issuer or not _G_ADDRESS_RE.match(asset_issuer.strip()):
            errs.append(
                FieldValidationError(
                    field="asset_issuer",
                    message="Non-native assets require a valid 56-character G... issuer address.",
                )
            )

    if tx_hash is not None and not _TX_HASH_RE.match(tx_hash.strip()):
        errs.append(
            FieldValidationError(
                field="tx_hash",
                message="tx_hash must be a 64-character hexadecimal string.",
            )
        )

    if amount is not None:
        try:
            dec = Decimal(amount.strip())
            if dec <= Decimal("0") or dec > Decimal("922337203685.4775807"):
                errs.append(
                    FieldValidationError(
                        field="amount",
                        message="amount must be positive and within Stellar int64 stroop bounds.",
                    )
                )
        except InvalidOperation:
            errs.append(
                FieldValidationError(
                    field="amount",
                    message="amount must be a valid decimal string.",
                )
            )

    if errs:
        return StellarBoundaryValidationResult(is_valid=False, status_code=422, errors=tuple(errs))
    return StellarBoundaryValidationResult(is_valid=True, status_code=200, errors=())
