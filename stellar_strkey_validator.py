"""Validate Stellar StrKeys and muxed account inputs (#557)."""

from __future__ import annotations

import base64
import binascii
from dataclasses import asdict, dataclass
import struct
from typing import Any

VERSION_BYTE_ACCOUNT_ID = 6 << 3  # 'G' (48)
VERSION_BYTE_MUXED_ACCOUNT = 12 << 3  # 'M' (96)
VERSION_BYTE_SECRET_SEED = 18 << 3  # 'S' (144)


def _crc16_xmodem(data: bytes) -> int:
    crc = 0x0000
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def encode_stellar_strkey(version_byte: int, payload: bytes) -> str:
    """Encode raw payload and version byte into canonical Stellar StrKey."""
    raw = bytes([version_byte]) + payload
    checksum = struct.pack("<H", _crc16_xmodem(raw))
    return base64.b32encode(raw + checksum).decode("ascii").rstrip("=")


@dataclass(frozen=True)
class StrKeyValidationResult:
    """Validation result for Stellar public and muxed account identifiers."""

    is_valid: bool
    address_kind: str | None
    error_code: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_stellar_account_strkey(
    address: str, *, allow_muxed: bool = True
) -> StrKeyValidationResult:
    """Validate checksum, version byte, and payload length without echoing sensitive inputs."""
    candidate = address.strip()
    if not candidate:
        return StrKeyValidationResult(
            is_valid=False,
            address_kind=None,
            error_code="EMPTY_ADDRESS",
            error_message="Stellar address cannot be empty.",
        )

    if candidate.startswith("S"):
        return StrKeyValidationResult(
            is_valid=False,
            address_kind=None,
            error_code="SECRET_KEY_FORBIDDEN",
            error_message="Secret seeds (S...) are never accepted by read-only tools.",
        )

    pad_len = (8 - (len(candidate) % 8)) % 8
    padded = candidate + ("=" * pad_len)
    try:
        decoded = base64.b32decode(padded, casefold=False)
    except (binascii.Error, ValueError):
        return StrKeyValidationResult(
            is_valid=False,
            address_kind=None,
            error_code="MALFORMED_STRKEY_ENCODING",
            error_message="Address is not valid uppercase Base32 StrKey encoding.",
        )

    if len(decoded) < 3:
        return StrKeyValidationResult(
            is_valid=False,
            address_kind=None,
            error_code="MALFORMED_STRKEY_LENGTH",
            error_message="Decoded StrKey is too short.",
        )

    body, checksum_bytes = decoded[:-2], decoded[-2:]
    expected_crc = struct.unpack("<H", checksum_bytes)[0]
    actual_crc = _crc16_xmodem(body)
    if expected_crc != actual_crc:
        return StrKeyValidationResult(
            is_valid=False,
            address_kind=None,
            error_code="INVALID_STRKEY_CHECKSUM",
            error_message="StrKey CRC16-XModem checksum verification failed.",
        )

    version_byte = body[0]
    payload = body[1:]

    if version_byte == VERSION_BYTE_SECRET_SEED:
        return StrKeyValidationResult(
            is_valid=False,
            address_kind=None,
            error_code="SECRET_KEY_FORBIDDEN",
            error_message="Secret seeds are forbidden.",
        )

    if (
        version_byte == VERSION_BYTE_ACCOUNT_ID
        and len(payload) == 32
        and len(candidate) == 56
    ):
        return StrKeyValidationResult(
            is_valid=True,
            address_kind="ed25519_public_key",
        )

    if (
        version_byte == VERSION_BYTE_MUXED_ACCOUNT
        and len(payload) == 40
        and len(candidate) == 69
    ):
        if not allow_muxed:
            return StrKeyValidationResult(
                is_valid=False,
                address_kind=None,
                error_code="MUXED_ACCOUNT_NOT_SUPPORTED",
                error_message="This endpoint requires a standard G... account, not a muxed M... account.",
            )
        return StrKeyValidationResult(
            is_valid=True,
            address_kind="med25519_muxed_account",
        )

    return StrKeyValidationResult(
        is_valid=False,
        address_kind=None,
        error_code="UNSUPPORTED_STRKEY_TYPE",
        error_message="Unsupported Stellar StrKey version or payload length.",
    )
