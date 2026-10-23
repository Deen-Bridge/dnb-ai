"""Tests for stable error codes, common error envelope, and transport mapping for Stellar network tools (#587)."""

from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from httpx import ASGITransport, AsyncClient
from stellar_sdk.exceptions import NotFoundError

import stellar
from stellar_error_codes import (
    CODE_INVALID_PARAMETER,
    CODE_LEDGER_STALE,
    CODE_PROVIDER_TIMEOUT,
    CODE_PROVIDER_UNAVAILABLE,
    CODE_UNSUPPORTED_NETWORK,
    ERROR_POLICIES,
    StellarErrorClass,
    build_stellar_tool_error,
    map_transport_exception_to_error_envelope,
)


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(stellar.router)

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        headers = getattr(exc, "headers", None) or {}
        hint = getattr(exc, "hint", None)
        if isinstance(exc.detail, dict):
            content = dict(exc.detail)
            if hint and "hint" not in content:
                content["hint"] = hint
        else:
            content = {"detail": exc.detail}
            if hint:
                content["hint"] = hint
        return JSONResponse(status_code=exc.status_code, content=jsonable_encoder(content), headers=headers)

    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


def test_all_five_error_classes_have_documented_policies() -> None:
    expected_classes = {
        StellarErrorClass.INVALID_INPUT,
        StellarErrorClass.UNAVAILABLE_PROVIDER,
        StellarErrorClass.NOT_FOUND,
        StellarErrorClass.STALE_DATA,
        StellarErrorClass.UNSUPPORTED_NETWORK,
    }
    assert set(ERROR_POLICIES.keys()) == expected_classes

    # Verify retryability and user actions
    assert ERROR_POLICIES[StellarErrorClass.INVALID_INPUT].retryable is False
    assert ERROR_POLICIES[StellarErrorClass.INVALID_INPUT].user_action == "change_request"

    assert ERROR_POLICIES[StellarErrorClass.UNAVAILABLE_PROVIDER].retryable is True
    assert ERROR_POLICIES[StellarErrorClass.UNAVAILABLE_PROVIDER].user_action == "retry_later"

    assert ERROR_POLICIES[StellarErrorClass.NOT_FOUND].retryable is False
    assert ERROR_POLICIES[StellarErrorClass.NOT_FOUND].user_action == "change_request"

    assert ERROR_POLICIES[StellarErrorClass.STALE_DATA].retryable is True
    assert ERROR_POLICIES[StellarErrorClass.STALE_DATA].user_action == "refresh_data"

    assert ERROR_POLICIES[StellarErrorClass.UNSUPPORTED_NETWORK].retryable is False
    assert ERROR_POLICIES[StellarErrorClass.UNSUPPORTED_NETWORK].user_action == "change_request"


def test_transport_exception_mapping_and_diagnostics_redaction() -> None:
    # 1. Timeout -> UNAVAILABLE_PROVIDER
    timeout_err = httpx.ReadTimeout("Read timed out on Horizon connection")
    env_timeout = map_transport_exception_to_error_envelope(timeout_err, network="testnet")
    assert env_timeout.error_class == StellarErrorClass.UNAVAILABLE_PROVIDER.value
    assert env_timeout.error_code == CODE_PROVIDER_TIMEOUT
    assert env_timeout.retryable is True

    # 2. Connect error -> UNAVAILABLE_PROVIDER
    connect_err = httpx.ConnectError("Connection refused by peer")
    env_connect = map_transport_exception_to_error_envelope(connect_err, network="public")
    assert env_connect.error_class == StellarErrorClass.UNAVAILABLE_PROVIDER.value
    assert env_connect.error_code == CODE_PROVIDER_UNAVAILABLE
    assert env_connect.retryable is True

    # 3. Not Found -> NOT_FOUND
    from unittest.mock import MagicMock

    mock_resp = MagicMock(status_code=404, text="Resource not found")
    mock_resp.json.return_value = {"detail": "Resource not found"}
    not_found_err = NotFoundError(mock_resp)
    env_not_found = map_transport_exception_to_error_envelope(not_found_err, network="testnet")
    assert env_not_found.error_class == StellarErrorClass.NOT_FOUND.value
    assert env_not_found.retryable is False

    # 4. Stale data -> STALE_DATA
    env_stale = map_transport_exception_to_error_envelope("Stale ledger sequence expired", network="testnet")
    assert env_stale.error_class == StellarErrorClass.STALE_DATA.value
    assert env_stale.error_code == CODE_LEDGER_STALE
    assert env_stale.retryable is True

    # 5. Unsupported network -> UNSUPPORTED_NETWORK
    env_net = map_transport_exception_to_error_envelope("Ok", network="futurenet-beta")
    assert env_net.error_class == StellarErrorClass.UNSUPPORTED_NETWORK.value
    assert env_net.error_code == CODE_UNSUPPORTED_NETWORK
    assert env_net.retryable is False

    # 6. Diagnostics redaction: raw addresses and secrets must be sanitized in server-side diagnostics
    secret_key = "SBBD47IF6LWK7P7MDEVSCWR7DPUWV3NY3DTQEVFL4NAT4AQH3ZLLFLA5"
    public_key = "GBBD47IF6LWK7P7MDEVSCWR7DPUWV3NY3DTQEVFL4NAT4AQH3ZLLFLA5"
    diag = {
        "account_id": public_key,
        "secret": secret_key,
        "memo": "Secret memo 123",
    }
    env_diag = build_stellar_tool_error(
        StellarErrorClass.INVALID_INPUT,
        error_code=CODE_INVALID_PARAMETER,
        diagnostics=diag,
    )
    assert env_diag.diagnostics is not None
    assert secret_key not in str(env_diag.diagnostics)
    assert env_diag.diagnostics is not None
    assert env_diag.diagnostics["secret"] == "[REDACTED]"
    assert env_diag.diagnostics["memo"] == "[REDACTED]"
    assert public_key not in str(env_diag.diagnostics)
    assert env_diag.diagnostics["account_id"].startswith("corr_")


@pytest.mark.asyncio
async def test_api_common_error_envelope_for_each_error_class(client):
    async with client:
        # Test 1: invalid_input (HTTP 400)
        resp1 = await client.post(
            "/stellar/tool/error-envelope",
            json={"error_class": "invalid_input", "raise_exception": True},
        )
        assert resp1.status_code == 400
        data1 = resp1.json()
        assert data1["error_class"] == "invalid_input"
        assert data1["retryable"] is False
        assert data1["user_action"] == "change_request"
        assert "hint" in data1

        # Test 2: unavailable_provider (HTTP 503)
        resp2 = await client.post(
            "/stellar/tool/error-envelope",
            json={"error_class": "unavailable_provider", "raise_exception": True},
        )
        assert resp2.status_code == 503
        data2 = resp2.json()
        assert data2["error_class"] == "unavailable_provider"
        assert data2["retryable"] is True
        assert data2["user_action"] == "retry_later"

        # Test 3: not_found (HTTP 404)
        resp3 = await client.post(
            "/stellar/tool/error-envelope",
            json={"error_class": "not_found", "raise_exception": True},
        )
        assert resp3.status_code == 404
        data3 = resp3.json()
        assert data3["error_class"] == "not_found"
        assert data3["retryable"] is False

        # Test 4: stale_data (HTTP 409)
        resp4 = await client.post(
            "/stellar/tool/error-envelope",
            json={"error_class": "stale_data", "raise_exception": True},
        )
        assert resp4.status_code == 409
        data4 = resp4.json()
        assert data4["error_class"] == "stale_data"
        assert data4["retryable"] is True
        assert data4["user_action"] == "refresh_data"

        # Test 5: unsupported_network (HTTP 400)
        resp5 = await client.post(
            "/stellar/tool/error-envelope",
            json={"error_class": "unsupported_network", "raise_exception": True},
        )
        assert resp5.status_code == 400
        data5 = resp5.json()
        assert data5["error_class"] == "unsupported_network"
        assert data5["retryable"] is False
        assert data5["user_action"] == "change_request"


@pytest.mark.asyncio
async def test_api_transport_error_mapping_endpoint(client):
    async with client:
        # Pass a timeout exception string
        resp = await client.post(
            "/stellar/tool/error-envelope",
            json={"transport_exception": "ReadTimeout: Horizon provider did not respond"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["error_class"] == "unavailable_provider"
        assert data["error_code"] == CODE_PROVIDER_TIMEOUT
        assert data["retryable"] is True
