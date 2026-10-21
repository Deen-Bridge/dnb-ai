"""Tests for Per-Request Token Budgeting and Context Compression (#92)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from main import app
from token_budget import (
    HistoryTurnInput,
    PromptIntent,
    RetrievedBlockInput,
    TokenBudgetRequest,
    enforce_token_budget,
)

client = TestClient(app)


def test_token_budget_never_exceeds_ceiling_and_prioritizes_high_salience() -> None:
    req = TokenBudgetRequest(
        query="What is the nisab of zakat on gold and silver?",
        intent=PromptIntent.zakat,
        input_token_ceiling=350,
        retrieved_blocks=[
            RetrievedBlockInput(
                block_id="low-salience",
                text=("Unrelated historical background paragraph. " * 40),
                salience=0.2,
            ),
            RetrievedBlockInput(
                block_id="high-salience",
                text=(
                    "The nisab of zakat on gold is 85 grams of pure gold, and on silver it is 595 grams. "
                    + ("Additional commentary on agricultural produce. " * 25)
                ),
                salience=0.95,
            ),
            RetrievedBlockInput(
                block_id="duplicate-high",
                text=(
                    "The nisab of zakat on gold is 85 grams of pure gold, and on silver it is 595 grams. "
                    + ("Additional commentary on agricultural produce. " * 25)
                ),
                salience=0.90,
            ),
        ],
        history=[
            HistoryTurnInput(role="user", content="Old turn " * 60),
            HistoryTurnInput(role="assistant", content="Old reply " * 60),
            HistoryTurnInput(role="user", content="Recent question about zakat calculation"),
        ],
    )
    res = enforce_token_budget(req)
    assert res.budgeted_input_tokens <= res.input_token_ceiling
    assert res.max_output_tokens == 1024
    assert "high-salience" in res.kept_block_ids
    assert "duplicate-high" in res.dropped_block_ids
    assert res.tokens_saved > 0
    assert res.reduction_percent > 30.0


def test_token_budget_endpoint() -> None:
    resp = client.post(
        "/context/token-budget/allocate",
        json={
            "query": "Explain Surah Al-Ikhlas",
            "intent": "tafsir",
            "input_token_ceiling": 400,
            "retrieved_blocks": [
                {
                    "block_id": "tafsir-112",
                    "text": "Surah Al-Ikhlas affirms pure Tawhid and the uniqueness of Allah.",
                    "salience": 0.9,
                }
            ],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["max_output_tokens"] == 2048
    assert data["budgeted_input_tokens"] <= 400
