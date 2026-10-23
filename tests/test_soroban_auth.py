"""Tests for the Soroban authorization entry safety explainer.

All tests are strictly offline and non-custodial. Secret keys and seed phrases
are verified to be detected, refused, and redacted, with zero network transmission.
"""

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from stellar_sdk import scval, xdr

import main
import stellar
from main import app
from soroban import (
    RiskLevel,
    SorobanAuthExplainRequest,
    build_chat_soroban_auth_context,
    contains_secret_key,
    contains_seed_phrase,
    contains_sensitive_key_material,
    explain_soroban_authorization,
    explorer_account_url,
    explorer_contract_url,
    is_soroban_auth_question,
    redact_sensitive_material,
)

client = TestClient(app)

SAMPLE_ACCOUNT = "GA5ZSEJYB37JRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN"
SAMPLE_RECIPIENT = "GBBD47IF6LWK7P7MDEVSCWR7DPUWV3NY3DTQEVFL4NAT4AQH3ZLLFLA5"
SAMPLE_CONTRACT = "CCYP5OSKQ4TCQJQE73G44KVBSTUT26POGHDNS23MMXUBPQRW54O43OB6"
SECRET_KEY = "S" + "A" * 55
SAMPLE_SEED_PHRASE = "abandon ability able about above absent absorb abstract absurd abuse access accident"


def run(coro):
    return asyncio.run(coro)


def build_test_entry_xdr(
    function_name: str = "transfer",
    args: list | None = None,
    contract_id: str = SAMPLE_CONTRACT,
    auth_address: str = SAMPLE_ACCOUNT,
    nonce: int = 1,
    expiration_ledger: int = 1000,
    sub_invocations: list | None = None,
    creds_type: str = "address",
) -> str:
    """Helper to generate valid Base64 SorobanAuthorizationEntry XDR for offline testing."""
    if args is None:
        args = [
            scval.to_address(auth_address),
            scval.to_address(SAMPLE_RECIPIENT),
            scval.to_int128(25000000),  # 2.5 USDC / units
        ]

    c_addr = scval.to_address(contract_id).address
    assert c_addr is not None

    inv = xdr.SorobanAuthorizedInvocation(
        function=xdr.SorobanAuthorizedFunction(
            type=xdr.SorobanAuthorizedFunctionType.SOROBAN_AUTHORIZED_FUNCTION_TYPE_CONTRACT_FN,
            contract_fn=xdr.InvokeContractArgs(
                contract_address=c_addr,
                function_name=xdr.SCSymbol(function_name.encode()),
                args=args,
            ),
        ),
        sub_invocations=sub_invocations or [],
    )

    if creds_type == "source_account":
        creds = xdr.SorobanCredentials(
            type=xdr.SorobanCredentialsType.SOROBAN_CREDENTIALS_SOURCE_ACCOUNT,
        )
    else:
        a_addr = scval.to_address(auth_address).address
        assert a_addr is not None
        creds = xdr.SorobanCredentials(
            type=xdr.SorobanCredentialsType.SOROBAN_CREDENTIALS_ADDRESS,
            address=xdr.SorobanAddressCredentials(
                address=a_addr,
                nonce=xdr.Int64(nonce),
                signature_expiration_ledger=xdr.Uint32(expiration_ledger),
                signature=scval.to_void(),
            ),
        )

    entry = xdr.SorobanAuthorizationEntry(credentials=creds, root_invocation=inv)
    return entry.to_xdr()


# ---------------------------------------------------------------------------
# 1. Detection Tests
# ---------------------------------------------------------------------------


class TestSorobanAuthDetection:
    @pytest.mark.parametrize(
        "prompt,expected",
        [
            ("Can you explain this Soroban authorization entry for me?", True),
            ("Is this Soroban auth entry safe to sign in Freighter?", True),
            (
                "What does this contract authorization do: CCYP5OSKQ4TCQJQE73G44KVBSTUT26POGHDNS23MMXUBPQRW54O43OB6?",
                True,
            ),
            ("simulateTransaction returned this auth entry, can you check it?", True),
            ("Is this InvokeHostFunction authorization safe?", True),
            ("How do I calculate zakat on my Stellar USDC?", False),
            ("Did my course payment go through?", False),
            ("What is the meaning of Surah Al-Fatiha?", False),
            ("", False),
        ],
    )
    def test_soroban_question_detection(self, prompt, expected):
        assert is_soroban_auth_question(prompt) is expected


# ---------------------------------------------------------------------------
# 2. Secret Key & Seed Phrase Non-Custodial Safety
# ---------------------------------------------------------------------------


class TestSecretKeyAndSeedPhraseProtection:
    def test_secret_key_detected(self):
        text = f"Here is my key: {SECRET_KEY} please explain this auth entry"
        assert contains_secret_key(text) is True
        assert contains_sensitive_key_material(text) is True

    def test_seed_phrase_detected(self):
        text = f"My wallet recovery phrase is {SAMPLE_SEED_PHRASE}"
        assert contains_seed_phrase(text) is True
        assert contains_sensitive_key_material(text) is True

    def test_sensitive_material_redacted(self):
        text = f"My secret key is {SECRET_KEY} and seed phrase: {SAMPLE_SEED_PHRASE}"
        redacted = redact_sensitive_material(text)
        assert redacted is not None
        assert SECRET_KEY not in redacted
        assert "[REDACTED STELLAR SECRET KEY]" in redacted

    def test_secret_key_in_request_triggers_critical_refusal(self):
        req = SorobanAuthExplainRequest(
            contract_id=SAMPLE_CONTRACT,
            function_name="transfer",
            args=[SECRET_KEY],
        )
        explanation = explain_soroban_authorization(req)
        assert explanation.risk_level == RiskLevel.CRITICAL
        assert explanation.secret_key_detected is True
        assert "Refused" in explanation.summary
        assert any("CRITICAL SECURITY ALERT" in w for w in explanation.warnings)

    def test_seed_phrase_in_request_triggers_critical_refusal(self):
        req = SorobanAuthExplainRequest(
            contract_id=SAMPLE_CONTRACT,
            function_name="transfer",
            args=[SAMPLE_SEED_PHRASE],
        )
        explanation = explain_soroban_authorization(req)
        assert explanation.risk_level == RiskLevel.CRITICAL
        assert explanation.secret_key_detected is True
        assert "Refused" in explanation.summary

    def test_chat_turn_refuses_when_secret_key_present(self):
        prompt = f"Explain this Soroban auth entry with secret {SECRET_KEY}"
        ctx = run(build_chat_soroban_auth_context(prompt))
        assert ctx is not None
        assert ctx.info.secret_key_detected is True
        assert ctx.info.explained is False
        assert "CRITICAL SECURITY ALERT" in ctx.prompt_block
        assert SECRET_KEY not in ctx.prompt_block


# ---------------------------------------------------------------------------
# 3. Explicit Parsing & Risk Analysis Tests
# ---------------------------------------------------------------------------


class TestSorobanXdrAnalysis:
    def test_standard_transfer_invocation(self):
        xdr_b64 = build_test_entry_xdr(
            function_name="transfer",
            contract_id=SAMPLE_CONTRACT,
            auth_address=SAMPLE_ACCOUNT,
            nonce=5,
            expiration_ledger=50000,
        )
        req = SorobanAuthExplainRequest(entry_xdr=xdr_b64)
        explanation = explain_soroban_authorization(req)

        assert explanation.risk_level == RiskLevel.CAUTION
        assert explanation.authorized_address == SAMPLE_ACCOUNT
        assert explanation.nonce == 5
        assert explanation.signature_expiration_ledger == 50000
        assert explanation.total_invocations_count == 1
        assert explanation.sub_invocations_count == 0
        assert len(explanation.transfers) == 1
        assert explanation.transfers[0].from_address == SAMPLE_ACCOUNT
        assert explanation.transfers[0].to_address == SAMPLE_RECIPIENT
        assert explanation.transfers[0].amount == "25000000"
        assert explanation.is_non_custodial is True
        assert explanation.root_invocation is not None
        assert explanation.root_invocation.contract_explorer is not None
        assert "stellar.expert" in explanation.root_invocation.contract_explorer

    def test_unlimited_allowance_approval(self):
        unlimited_amount = 2**127 - 1
        args = [
            scval.to_address(SAMPLE_ACCOUNT),
            scval.to_address(SAMPLE_RECIPIENT),
            scval.to_int128(unlimited_amount),
            scval.to_uint32(200000),
        ]
        xdr_b64 = build_test_entry_xdr(function_name="approve", args=args)
        req = SorobanAuthExplainRequest(entry_xdr=xdr_b64)
        explanation = explain_soroban_authorization(req)

        assert explanation.risk_level == RiskLevel.WARNING
        assert len(explanation.allowances) == 1
        assert explanation.allowances[0].is_unlimited is True
        assert any("Unlimited" in w or "unlimited" in w for w in explanation.warnings)

    def test_admin_privilege_invocation(self):
        args = [scval.to_address(SAMPLE_RECIPIENT)]
        xdr_b64 = build_test_entry_xdr(function_name="set_admin", args=args)
        req = SorobanAuthExplainRequest(entry_xdr=xdr_b64)
        explanation = explain_soroban_authorization(req)

        assert explanation.risk_level == RiskLevel.WARNING
        assert explanation.root_invocation is not None
        assert any("Administrative" in r for r in explanation.root_invocation.risk_factors)

    def test_zero_expiration_ledger_warning(self):
        xdr_b64 = build_test_entry_xdr(expiration_ledger=0)
        req = SorobanAuthExplainRequest(entry_xdr=xdr_b64)
        explanation = explain_soroban_authorization(req)

        assert any("expiration ledger is 0" in w for w in explanation.warnings)


# ---------------------------------------------------------------------------
# 4. Sub-Invocation Tree Analysis (Regression test for hidden sub-calls)
# ---------------------------------------------------------------------------


class TestSubInvocationsAnalysis:
    def test_nested_sub_invocation_transfer_detected(self):
        # Child sub-invocation that silently transfers tokens
        child_addr = scval.to_address(SAMPLE_CONTRACT).address
        assert child_addr is not None
        child_inv = xdr.SorobanAuthorizedInvocation(
            function=xdr.SorobanAuthorizedFunction(
                type=xdr.SorobanAuthorizedFunctionType.SOROBAN_AUTHORIZED_FUNCTION_TYPE_CONTRACT_FN,
                contract_fn=xdr.InvokeContractArgs(
                    contract_address=child_addr,
                    function_name=xdr.SCSymbol(b"transfer"),
                    args=[
                        scval.to_address(SAMPLE_ACCOUNT),
                        scval.to_address(SAMPLE_RECIPIENT),
                        scval.to_int128(99999999),
                    ],
                ),
            ),
            sub_invocations=[],
        )

        # Root invocation: innocuous "swap" or "deposit"
        root_xdr = build_test_entry_xdr(
            function_name="deposit",
            args=[scval.to_uint32(100)],
            sub_invocations=[child_inv],
        )

        req = SorobanAuthExplainRequest(entry_xdr=root_xdr)
        explanation = explain_soroban_authorization(req)

        assert explanation.total_invocations_count == 2
        assert explanation.sub_invocations_count == 1
        assert explanation.risk_level == RiskLevel.WARNING
        assert len(explanation.transfers) == 1
        assert explanation.transfers[0].is_sub_invocation is True
        assert any("Sub-invocation transfer" in w for w in explanation.warnings)


# ---------------------------------------------------------------------------
# 5. Network Awareness & Explorer URLs
# ---------------------------------------------------------------------------


class TestNetworkAwareness:
    def test_explorer_urls_respect_network(self):
        assert (
            explorer_contract_url(SAMPLE_CONTRACT, "testnet")
            == f"https://stellar.expert/explorer/testnet/contract/{SAMPLE_CONTRACT}"
        )
        assert (
            explorer_contract_url(SAMPLE_CONTRACT, "public")
            == f"https://stellar.expert/explorer/public/contract/{SAMPLE_CONTRACT}"
        )
        assert (
            explorer_account_url(SAMPLE_ACCOUNT, "testnet")
            == f"https://stellar.expert/explorer/testnet/account/{SAMPLE_ACCOUNT}"
        )
        assert (
            explorer_account_url(SAMPLE_ACCOUNT, "public")
            == f"https://stellar.expert/explorer/public/account/{SAMPLE_ACCOUNT}"
        )

    def test_explanation_uses_configured_network(self):
        xdr_b64 = build_test_entry_xdr()
        req_testnet = SorobanAuthExplainRequest(entry_xdr=xdr_b64, network="testnet")
        exp_testnet = explain_soroban_authorization(req_testnet)
        assert exp_testnet.network == "testnet"
        assert exp_testnet.root_invocation is not None
        assert exp_testnet.root_invocation.contract_explorer is not None
        assert "explorer/testnet" in exp_testnet.root_invocation.contract_explorer

        req_public = SorobanAuthExplainRequest(entry_xdr=xdr_b64, network="public")
        exp_public = explain_soroban_authorization(req_public)
        assert exp_public.network == "public"
        assert exp_public.root_invocation is not None
        assert exp_public.root_invocation.contract_explorer is not None
        assert "explorer/public" in exp_public.root_invocation.contract_explorer


# ---------------------------------------------------------------------------
# 6. Credentials Types
# ---------------------------------------------------------------------------


class TestCredentialsTypes:
    def test_source_account_credentials(self):
        xdr_b64 = build_test_entry_xdr(creds_type="source_account")
        req = SorobanAuthExplainRequest(entry_xdr=xdr_b64)
        explanation = explain_soroban_authorization(req)

        assert explanation.credentials_type == "source_account"
        assert explanation.authorized_address is None
        assert explanation.nonce is None
        assert explanation.signature_expiration_ledger is None


# ---------------------------------------------------------------------------
# 7. FastAPI Endpoint Tests
# ---------------------------------------------------------------------------


class TestSorobanApiEndpoints:
    def test_explain_endpoint_with_valid_xdr(self):
        xdr_b64 = build_test_entry_xdr()
        response = client.post("/stellar/soroban/auth/explain", json={"entry_xdr": xdr_b64})
        assert response.status_code == 200
        data = response.json()
        assert data["risk_level"] in ("safe", "caution", "warning")
        assert data["is_non_custodial"] is True
        assert data["authorized_address"] == SAMPLE_ACCOUNT

    def test_explain_endpoint_alias(self):
        xdr_b64 = build_test_entry_xdr()
        response = client.post("/stellar/soroban-auth-explain", json={"entry_xdr": xdr_b64})
        assert response.status_code == 200
        data = response.json()
        assert data["total_invocations_count"] == 1

    def test_explain_endpoint_with_invalid_xdr(self):
        response = client.post("/stellar/soroban/auth/explain", json={"entry_xdr": "not-valid-base64-xdr"})
        assert response.status_code == 400
        assert "Invalid Soroban authorization entry XDR" in response.json()["detail"]

    def test_explain_endpoint_with_direct_fields(self):
        response = client.post(
            "/stellar/soroban/auth/explain",
            json={
                "contract_id": SAMPLE_CONTRACT,
                "function_name": "transfer",
                "args": [SAMPLE_ACCOUNT, SAMPLE_RECIPIENT, "1000000"],
                "auth_address": SAMPLE_ACCOUNT,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["root_invocation"]["function_name"] == "transfer"
        assert len(data["transfers"]) == 1

    def test_stellar_info_lists_features(self):
        response = client.get("/stellar/info")
        assert response.status_code == 200
        features = response.json()["features"]
        assert "soroban-auth-explainer" in features
        assert "chat-soroban-auth" in features


# ---------------------------------------------------------------------------
# 8. Chat Context & System Prompt Grounding
# ---------------------------------------------------------------------------


@dataclass
class _FakePart:
    text: str


@dataclass
class _FakeContent:
    role: str
    text: str

    @property
    def parts(self) -> list[_FakePart]:
        return [_FakePart(self.text)]


class _FakeChatSession:
    def __init__(self) -> None:
        self.history: list[_FakeContent] = []
        self.messages: list[str] = []

    async def send_message_async(self, message: str, **kwargs) -> SimpleNamespace:
        self.messages.append(message)
        answer = "This Soroban authorization entry has caution risk."
        self.history.extend(
            [
                _FakeContent(role="user", text=message),
                _FakeContent(role="model", text=answer),
            ]
        )
        return SimpleNamespace(
            text=answer,
            candidates=[SimpleNamespace(finish_reason="STOP")],
            prompt_feedback=None,
        )


class _FakeModel:
    def __init__(self) -> None:
        self.sessions: list[_FakeChatSession] = []

    def start_chat(self, history=None) -> _FakeChatSession:
        session = _FakeChatSession()
        self.sessions.append(session)
        return session


class TestChatContextIntegration:
    def test_chat_context_with_xdr(self):
        xdr_b64 = build_test_entry_xdr()
        prompt = f"Please explain this Soroban auth entry: {xdr_b64}"
        ctx = run(build_chat_soroban_auth_context(prompt))
        assert ctx is not None
        assert ctx.info.explained is True
        assert ctx.info.contract_id == SAMPLE_CONTRACT
        assert "SOROBAN AUTHORIZATION SAFETY ANALYSIS" in ctx.prompt_block
        assert "RULES FOR THE ASSISTANT:" in ctx.prompt_block

    def test_chat_context_educational_when_no_xdr(self):
        prompt = "How does Soroban smart contract authorization work and what should I check?"
        ctx = run(build_chat_soroban_auth_context(prompt))
        assert ctx is not None
        assert ctx.info.explained is False
        assert "SOROBAN AUTHORIZATION SAFETY QUESTION" in ctx.prompt_block
        assert "Never enter or expose secret keys or seed phrases" in ctx.prompt_block

    def test_chat_api_includes_soroban_auth_info(self, monkeypatch):
        fake_model = _FakeModel()
        monkeypatch.setattr(main, "get_model", lambda: fake_model)
        monkeypatch.setenv("SAFETY_PIPELINE_ENABLED", "false")
        monkeypatch.setattr(main, "SEMANTIC_CACHE_ENABLED", False)
        monkeypatch.setattr(main, "active_chats", {})

        xdr_b64 = build_test_entry_xdr()
        with patch.object(stellar, "STELLAR_NETWORK", "testnet"):
            response = client.post(
                "/chat",
                json={
                    "prompt": "Is this Soroban authorization entry safe?",
                    "soroban_auth_entry": xdr_b64,
                },
            )
            assert response.status_code == 200
            data = response.json()
            assert "soroban_auth" in data
            assert data["soroban_auth"] is not None
            assert data["soroban_auth"]["explained"] is True
            assert data["soroban_auth"]["contract_id"] == SAMPLE_CONTRACT
