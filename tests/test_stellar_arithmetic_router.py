"""Tests for routing Stellar arithmetic to deterministic tools (#583)."""

from __future__ import annotations

from stellar_arithmetic_router import execute_deterministic_stellar_math


def test_deterministic_stroops_and_amount_addition() -> None:
    s = execute_deterministic_stellar_math(operation="xlm_to_stroops", values=["2.5"])
    assert s.exact_numeric_output == "25000000"
    assert "25000000" in s.explanation

    x = execute_deterministic_stellar_math(operation="stroops_to_xlm", values=["100"])
    assert x.exact_numeric_output == "0.0000100"

    total = execute_deterministic_stellar_math(operation="add_amounts", values=["10.1234567", "5.0000003"])
    assert total.exact_numeric_output == "15.1234570"
