"""Tests for Stellar StrKey and muxed account validator (#557)."""

from __future__ import annotations

from stellar_strkey_validator import (
    VERSION_BYTE_ACCOUNT_ID,
    VERSION_BYTE_MUXED_ACCOUNT,
    VERSION_BYTE_SECRET_SEED,
    encode_stellar_strkey,
    validate_stellar_account_strkey,
)


def test_valid_g_and_m_addresses_and_checksum_validation() -> None:
    valid_g = encode_stellar_strkey(VERSION_BYTE_ACCOUNT_ID, b"\x11" * 32)
    valid_m = encode_stellar_strkey(VERSION_BYTE_MUXED_ACCOUNT, b"\x22" * 40)
    secret_s = encode_stellar_strkey(VERSION_BYTE_SECRET_SEED, b"\x33" * 32)

    g_res = validate_stellar_account_strkey(valid_g)
    assert g_res.is_valid is True
    assert g_res.address_kind == "ed25519_public_key"

    m_res = validate_stellar_account_strkey(valid_m, allow_muxed=True)
    assert m_res.is_valid is True
    assert m_res.address_kind == "med25519_muxed_account"

    m_disallowed = validate_stellar_account_strkey(valid_m, allow_muxed=False)
    assert m_disallowed.is_valid is False
    assert m_disallowed.error_code == "MUXED_ACCOUNT_NOT_SUPPORTED"

    s_res = validate_stellar_account_strkey(secret_s)
    assert s_res.is_valid is False
    assert s_res.error_code == "SECRET_KEY_FORBIDDEN"
    assert secret_s not in (s_res.error_message or "")

    bad_crc = valid_g[:-1] + ("B" if valid_g[-1] != "B" else "C")
    crc_res = validate_stellar_account_strkey(bad_crc)
    assert crc_res.is_valid is False
    assert crc_res.error_code == "INVALID_STRKEY_CHECKSUM"
