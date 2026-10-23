"""Explain SEP-38 anchor RFQ quote responses without collecting secrets (#643).

SEP-38 (Anchor RFQ API) defines standard endpoints for anchors to provide firm and
indicative quotes for trading off-chain (fiat) and on-chain (Stellar) assets.
This module provides a non-custodial, network-aware explainer for SEP-38 quote
responses, interpreting trade directions, fee breakdowns, price metrics, expiration
windows, and execution steps while strictly rejecting secret seeds and seed phrases.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

OFFICIAL_SEP38_DOC_URL = "https://github.com/stellar/stellar-protocol/blob/master/ecosystem/sep-0038.md"

_SECRET_SEED_PATTERN = re.compile(r"\bS[A-Z2-7]{55}\b")
_BIP39_MNEMONIC_PATTERN = re.compile(
    r"\b(?:[a-z]{3,12}\s+){11,23}[a-z]{3,12}\b",
    re.IGNORECASE,
)

NON_CUSTODIAL_SAFETY_POLICY = (
    "SEP-38 RFQ quotes are commercial rate commitments provided by independent anchors. "
    "The AI assistant, Deen Bridge, and the Stellar network never request, receive, or store "
    "secret keys (S...) or recovery seed phrases. Any required transactions must be signed "
    "locally inside your non-custodial wallet."
)


def contains_secret_input(data: Any) -> bool:
    """Recursively inspect user input or payloads for secret keys or seed phrases."""
    if data is None:
        return False
    if isinstance(data, str):
        if _SECRET_SEED_PATTERN.search(data):
            return True
        # Check for 12 or 24-word seed phrase pattern
        cleaned = " ".join(data.strip().split())
        words = cleaned.split()
        if len(words) in (12, 15, 18, 21, 24) and all(w.isalpha() and w.islower() for w in words):
            return True
        if _BIP39_MNEMONIC_PATTERN.search(data):
            return True
        return False
    if isinstance(data, dict):
        return any(contains_secret_input(k) or contains_secret_input(v) for k, v in data.items())
    if isinstance(data, (list, tuple, set)):
        return any(contains_secret_input(item) for item in data)
    return False


@dataclass(frozen=True)
class Sep38AssetInfo:
    """Parsed representation of an asset identified under the SEP-38 scheme."""

    asset_type: str  # "iso4217", "stellar_native", "stellar_issued", "stellar_contract", "unknown"
    identifier: str
    code: str
    issuer: str | None
    is_onchain_stellar: bool
    explorer_url: str | None
    description: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Sep38FeeDetail:
    """Individual component within a SEP-38 fee breakdown."""

    name: str
    amount: str
    description: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Sep38FeeBreakdown:
    """Total fees and itemized breakdown for a SEP-38 quote."""

    total: str
    asset: str
    percentage_of_sell_amount: str | None
    percentage_of_buy_amount: str | None
    details: tuple[Sep38FeeDetail, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Sep38QuoteExplanation:
    """Comprehensive, non-custodial explanation of a SEP-38 RFQ quote response."""

    status: str  # "ok" or "rejected_secret_input"
    network: str
    quote_id: str | None
    flow_type: str  # "on_ramp", "off_ramp", "stellar_cross_asset", "off_chain_exchange", "unknown"
    sell_asset: Sep38AssetInfo | None
    sell_amount: str | None
    buy_asset: Sep38AssetInfo | None
    buy_amount: str | None
    price: str | None
    total_price: str | None
    rate_summary: str | None
    fee: Sep38FeeBreakdown | None
    expires_at: str | None
    is_expired: bool | None
    seconds_until_expiration: int | None
    is_firm_quote: bool
    execution_steps: tuple[str, ...]
    safety_policy: str
    official_sep_url: str
    rejection_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Sep38ProtocolGuide:
    """Overview of SEP-38 Anchor RFQ API capabilities and safety rules."""

    status: str  # "ok" or "rejected_secret_input"
    summary: str
    supported_endpoints: tuple[str, ...]
    quote_lifecycle_steps: tuple[str, ...]
    safety_policy: str
    official_sep_url: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_sep38_asset(asset_identifier: str, network: str = "public") -> Sep38AssetInfo:
    """Parse a SEP-38 asset identification string into a structured descriptor."""
    raw = asset_identifier.strip()
    net = network.strip().lower()

    if raw.startswith("iso4217:"):
        code = raw[len("iso4217:") :].strip().upper()
        return Sep38AssetInfo(
            asset_type="iso4217",
            identifier=raw,
            code=code,
            issuer=None,
            is_onchain_stellar=False,
            explorer_url=None,
            description=f"Off-chain fiat currency ({code}) settled via anchor banking rails.",
        )

    if raw in ("stellar:native", "stellar:XLM"):
        return Sep38AssetInfo(
            asset_type="stellar_native",
            identifier=raw,
            code="XLM",
            issuer=None,
            is_onchain_stellar=True,
            explorer_url=f"https://stellar.expert/explorer/{net}/asset/XLM",
            description=f"Native Stellar network lumen (XLM) on {net}.",
        )

    if raw.startswith("stellar:"):
        parts = raw[len("stellar:") :].split(":")
        if len(parts) == 2:
            code = parts[0].strip().upper()
            issuer = parts[1].strip()
            return Sep38AssetInfo(
                asset_type="stellar_issued",
                identifier=raw,
                code=code,
                issuer=issuer,
                is_onchain_stellar=True,
                explorer_url=f"https://stellar.expert/explorer/{net}/asset/{code}-{issuer}",
                description=f"On-chain issued Stellar asset {code} from issuer {issuer[:8]}... on {net}.",
            )
        if len(parts) == 1 and len(parts[0]) == 56 and parts[0].startswith("C"):
            contract_id = parts[0].strip()
            return Sep38AssetInfo(
                asset_type="stellar_contract",
                identifier=raw,
                code=contract_id,
                issuer=contract_id,
                is_onchain_stellar=True,
                explorer_url=f"https://stellar.expert/explorer/{net}/contract/{contract_id}",
                description=f"Soroban token contract {contract_id[:8]}... on {net}.",
            )

    return Sep38AssetInfo(
        asset_type="unknown",
        identifier=raw,
        code=raw,
        issuer=None,
        is_onchain_stellar=False,
        explorer_url=None,
        description=f"Unrecognized asset format: {raw}.",
    )


def _compute_percentage(amount_str: str | None, base_str: str | None) -> str | None:
    """Calculate percentage of amount over base with 2 decimal places."""
    if not amount_str or not base_str:
        return None
    try:
        amt = Decimal(amount_str)
        base = Decimal(base_str)
        if base <= Decimal("0"):
            return None
        pct = (amt / base) * Decimal("100")
        return f"{pct:.2f}%"
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None


def _determine_flow_type(sell_asset: Sep38AssetInfo, buy_asset: Sep38AssetInfo) -> str:
    """Classify the quote flow based on source and destination assets."""
    if not sell_asset.is_onchain_stellar and buy_asset.is_onchain_stellar:
        return "on_ramp"
    if sell_asset.is_onchain_stellar and not buy_asset.is_onchain_stellar:
        return "off_ramp"
    if sell_asset.is_onchain_stellar and buy_asset.is_onchain_stellar:
        return "stellar_cross_asset"
    if not sell_asset.is_onchain_stellar and not buy_asset.is_onchain_stellar:
        return "off_chain_exchange"
    return "unknown"


def _build_rate_summary(
    sell_code: str,
    sell_amt_str: str | None,
    buy_code: str,
    buy_amt_str: str | None,
    price_str: str | None,
    total_price_str: str | None,
) -> str | None:
    """Build an intuitive rate summary describing unit conversions and fee spread."""
    parts: list[str] = []
    if sell_amt_str and buy_amt_str:
        try:
            sell_amt = Decimal(sell_amt_str)
            buy_amt = Decimal(buy_amt_str)
            if sell_amt > 0 and buy_amt > 0:
                rate_buy_per_sell = (buy_amt / sell_amt).normalize()
                parts.append(f"Effective rate: 1 {sell_code} ≈ {rate_buy_per_sell} {buy_code}")
        except (InvalidOperation, ValueError):
            pass

    if total_price_str:
        parts.append(f"Total price (including fees): {total_price_str}")
    if price_str and price_str != total_price_str:
        parts.append(f"Base price (excluding fees): {price_str}")

    return " | ".join(parts) if parts else None


def _generate_execution_steps(
    flow_type: str,
    sell_asset: Sep38AssetInfo,
    buy_asset: Sep38AssetInfo,
    quote_id: str | None,
    network: str,
    anchor_domain: str | None,
    is_expired: bool | None,
) -> tuple[str, ...]:
    """Generate sequential execution guidance based on the quote flow."""
    anchor = anchor_domain or "the anchor service"
    id_ref = f"quote ID '{quote_id}'" if quote_id else "this quote"

    if is_expired:
        return (
            f"1. Quote Expired: This quote has expired and cannot be accepted by {anchor}.",
            "2. Request New Quote: Request a fresh quote from the anchor to establish a current locked price.",
            "3. Do Not Send Funds: Do not send funds against an expired quote reference.",
        )

    if flow_type == "on_ramp":
        return (
            f"1. Review Terms: Verify the locked exchange rate and ensure you have a trustline for {buy_asset.code} on Stellar {network}.",
            f"2. Initiate Deposit: Submit {id_ref} to {anchor}'s interactive or programmatic deposit flow (e.g. SEP-6 or SEP-24).",
            f"3. Deliver Off-Chain Funds: Transfer {sell_asset.code} to {anchor}'s payment instructions before the quote expires.",
            f"4. Receive On-Chain Settlement: {anchor} mints or delivers {buy_asset.code} directly to your Stellar account.",
        )

    if flow_type == "off_ramp":
        return (
            f"1. Review Terms: Confirm the destination payout details and the net {buy_asset.code} amount to be received.",
            f"2. Submit Withdrawal: Provide {id_ref} to {anchor}'s withdrawal endpoint (SEP-6, SEP-24, or SEP-31).",
            f"3. Sign On-Chain Payment: Use your non-custodial wallet to sign and send the {sell_asset.code} payment to {anchor}'s specified Stellar address with any required memo.",
            f"4. Payout Delivery: {anchor} detects the on-chain settlement and disburses {buy_asset.code} to your payout account.",
        )

    if flow_type == "stellar_cross_asset":
        return (
            f"1. Review Swap: Confirm the exchange from {sell_asset.code} to {buy_asset.code} on Stellar {network}.",
            f"2. Trustline Check: Ensure your wallet has an established trustline for receiving {buy_asset.code}.",
            f"3. Execute On-Chain: Submit {id_ref} to {anchor} and sign the required Stellar transaction using your local wallet.",
            f"4. Settle: Both legs settle on the Stellar {network} ledger.",
        )

    return (
        f"1. Review Details: Confirm amounts and fees with {anchor}.",
        f"2. Submit Quote: Provide {id_ref} to the anchor's API before the quote expiration time.",
        "3. Complete Transfer: Execute according to the anchor's protocol instructions without sharing private keys.",
    )


def explain_sep38_quote(
    quote: dict[str, Any] | str,
    *,
    network: str = "public",
    anchor_domain: str | None = None,
    now_utc: datetime | None = None,
) -> Sep38QuoteExplanation:
    """Analyze and explain a SEP-38 RFQ quote response while strictly rejecting secrets.

    Args:
        quote: Dict or JSON string of a SEP-38 quote response (e.g. from POST /quote or GET /quote/{id}).
        network: "public" or "testnet" (default: "public").
        anchor_domain: Optional domain of the anchor providing the quote.
        now_utc: Optional reference datetime for deterministic expiration checking (defaults to current UTC).

    Returns:
        Sep38QuoteExplanation detailing assets, prices, fee breakdown, expiration, and non-custodial guidance.
    """
    net = network.strip().lower()
    if net not in ("public", "testnet"):
        net = "public"

    # 1. Secret Protection
    if contains_secret_input(quote) or contains_secret_input(anchor_domain):
        return Sep38QuoteExplanation(
            status="rejected_secret_input",
            network=net,
            quote_id=None,
            flow_type="unknown",
            sell_asset=None,
            sell_amount=None,
            buy_asset=None,
            buy_amount=None,
            price=None,
            total_price=None,
            rate_summary=None,
            fee=None,
            expires_at=None,
            is_expired=None,
            seconds_until_expiration=None,
            is_firm_quote=False,
            execution_steps=(),
            safety_policy=NON_CUSTODIAL_SAFETY_POLICY,
            official_sep_url=OFFICIAL_SEP38_DOC_URL,
            rejection_reason=(
                "Rejected: Payload contains a Stellar secret seed (S...) or seed phrase. "
                "Never expose secret keys or recovery mnemonics to any service or AI assistant. "
                "SEP-38 RFQ operations are non-custodial and transaction signing must remain local to your wallet."
            ),
        )

    # 2. Parse payload
    if isinstance(quote, str):
        try:
            data: dict[str, Any] = json.loads(quote)
        except Exception:
            data = {}
    elif isinstance(quote, dict):
        data = quote
    else:
        data = {}

    quote_id = str(data.get("id")).strip() if data.get("id") else None
    sell_asset_raw = str(data.get("sell_asset", "")).strip()
    buy_asset_raw = str(data.get("buy_asset", "")).strip()
    sell_amount = str(data.get("sell_amount")).strip() if data.get("sell_amount") is not None else None
    buy_amount = str(data.get("buy_amount")).strip() if data.get("buy_amount") is not None else None
    price = str(data.get("price")).strip() if data.get("price") is not None else None
    total_price = str(data.get("total_price")).strip() if data.get("total_price") is not None else None
    expires_at_raw = str(data.get("expires_at")).strip() if data.get("expires_at") else None

    sell_info = parse_sep38_asset(sell_asset_raw, network=net) if sell_asset_raw else None
    buy_info = parse_sep38_asset(buy_asset_raw, network=net) if buy_asset_raw else None

    flow_type = _determine_flow_type(sell_info, buy_info) if sell_info and buy_info else "unknown"

    # 3. Expiration parsing
    is_expired: bool | None = None
    seconds_remaining: int | None = None
    if expires_at_raw:
        try:
            # Handle ISO-8601 with Z or timezone offset
            clean_ts = expires_at_raw.replace("Z", "+00:00")
            exp_dt = datetime.fromisoformat(clean_ts)
            current_time = now_utc if now_utc is not None else datetime.now(UTC)
            # Ensure exp_dt has timezone
            if exp_dt.tzinfo is None:
                exp_dt = exp_dt.replace(tzinfo=UTC)
            diff = (exp_dt - current_time).total_seconds()
            is_expired = diff <= 0
            seconds_remaining = max(0, int(diff))
        except (ValueError, TypeError):
            is_expired = None
            seconds_remaining = None

    # Firm quote: presence of quote ID and expiration timestamp guarantees price lock
    is_firm_quote = bool(quote_id and expires_at_raw)

    # 4. Fee breakdown parsing
    fee_obj = data.get("fee")
    fee_summary: Sep38FeeBreakdown | None = None
    if isinstance(fee_obj, dict):
        total_fee = str(fee_obj.get("total", "0")).strip()
        fee_asset = str(fee_obj.get("asset", "")).strip()

        details_list = fee_obj.get("details") or []
        details: list[Sep38FeeDetail] = []
        if isinstance(details_list, list):
            for d in details_list:
                if isinstance(d, dict):
                    details.append(
                        Sep38FeeDetail(
                            name=str(d.get("name", "Fee")).strip(),
                            amount=str(d.get("amount", "0")).strip(),
                            description=str(d["description"]).strip() if d.get("description") else None,
                        )
                    )

        pct_sell = _compute_percentage(total_fee, sell_amount) if fee_asset == sell_asset_raw else None
        pct_buy = _compute_percentage(total_fee, buy_amount) if fee_asset == buy_asset_raw else None

        fee_summary = Sep38FeeBreakdown(
            total=total_fee,
            asset=fee_asset,
            percentage_of_sell_amount=pct_sell,
            percentage_of_buy_amount=pct_buy,
            details=tuple(details),
        )

    rate_summary = _build_rate_summary(
        sell_info.code if sell_info else "SELL",
        sell_amount,
        buy_info.code if buy_info else "BUY",
        buy_amount,
        price,
        total_price,
    )

    steps = _generate_execution_steps(
        flow_type=flow_type,
        sell_asset=sell_info or parse_sep38_asset("unknown", network=net),
        buy_asset=buy_info or parse_sep38_asset("unknown", network=net),
        quote_id=quote_id,
        network=net,
        anchor_domain=anchor_domain,
        is_expired=is_expired,
    )

    return Sep38QuoteExplanation(
        status="ok",
        network=net,
        quote_id=quote_id,
        flow_type=flow_type,
        sell_asset=sell_info,
        sell_amount=sell_amount,
        buy_asset=buy_info,
        buy_amount=buy_amount,
        price=price,
        total_price=total_price,
        rate_summary=rate_summary,
        fee=fee_summary,
        expires_at=expires_at_raw,
        is_expired=is_expired,
        seconds_until_expiration=seconds_remaining,
        is_firm_quote=is_firm_quote,
        execution_steps=steps,
        safety_policy=NON_CUSTODIAL_SAFETY_POLICY,
        official_sep_url=OFFICIAL_SEP38_DOC_URL,
        rejection_reason=None,
    )


def explain_sep38_protocol(user_input: str | None = None) -> Sep38ProtocolGuide:
    """Provide a high-level explanation of the SEP-38 Anchor RFQ protocol.

    Rejects any secret seeds or seed phrases submitted in user_input.
    """
    if user_input is not None and contains_secret_input(user_input):
        return Sep38ProtocolGuide(
            status="rejected_secret_input",
            summary="Rejected: User prompt contains a secret seed or seed phrase.",
            supported_endpoints=(),
            quote_lifecycle_steps=(),
            safety_policy=NON_CUSTODIAL_SAFETY_POLICY,
            official_sep_url=OFFICIAL_SEP38_DOC_URL,
        )

    return Sep38ProtocolGuide(
        status="ok",
        summary=(
            "SEP-38 (Anchor RFQ API) enables wallets to request indicative prices and firm quotes "
            "from anchors for exchanging off-chain fiat currencies and on-chain Stellar assets."
        ),
        supported_endpoints=(
            "GET /info: Lists available assets, exchange pairs, and fee configurations.",
            "GET /prices: Returns indicative conversion rates for available asset pairs.",
            "GET /price: Returns an indicative quote for a specific amount and asset pair.",
            "POST /quote: Requests a firm, time-bounded quote with a guaranteed price and quote ID.",
            "GET /quote/{id}: Fetches the details and expiration state of an existing firm quote.",
        ),
        quote_lifecycle_steps=(
            "1. Discovery: Query anchor's /info to inspect supported assets and trading pairs.",
            "2. Price Discovery: Check /price for indicative rates without committing to a trade.",
            "3. Firm Quote Request: Submit authenticated POST /quote to lock in guaranteed pricing for a bounded duration.",
            "4. Protocol Handoff: Reference the quote ID in SEP-6, SEP-24, or SEP-31 transactions before expiration.",
            "5. Local Signing: Wallet signs on-chain transactions locally without exposing private keys.",
        ),
        safety_policy=NON_CUSTODIAL_SAFETY_POLICY,
        official_sep_url=OFFICIAL_SEP38_DOC_URL,
    )
