"""Offline tests for Stellar USDC conversion and chat calculations."""

import asyncio
from decimal import Decimal

import pytest
from fastapi import HTTPException

from stellar import (
    UsdcConversionRequest,
    build_chat_stellar_calculation_context,
    convert_usdc_units,
)


def run(coro):
    return asyncio.run(coro)


def test_convert_usdc_to_stroops_exactly():
    result = run(convert_usdc_units(UsdcConversionRequest(amount=Decimal("12.5"), direction="usdc_to_stroops")))
    assert result.output_amount == "125000000"
    assert "12.5000000 USDC" in result.explanation


def test_convert_stroops_to_usdc_exactly():
    result = run(convert_usdc_units(UsdcConversionRequest(amount=Decimal("1234567"), direction="stroops_to_usdc")))
    assert result.output_amount == "0.1234567"


def test_conversion_rejects_more_than_seven_decimal_places():
    with pytest.raises(HTTPException, match="at most 7 decimal places"):
        run(convert_usdc_units(UsdcConversionRequest(amount=Decimal("1.00000001"), direction="usdc_to_stroops")))


@pytest.mark.parametrize(
    "prompt,expected",
    [
        ("How many stroops are in 12.5 USDC?", "12.5000000 USDC = 125000000 stroops"),
        ("Convert 125000000 stroops to USDC", "125000000 stroops = 12.5000000 USDC"),
        ("What is 3% of 80 USDC?", "3% of 80.0000000 USDC is 2.4000000 USDC"),
        ("What is 80 USDC after 3% fee?", "the fee is 2.4000000 USDC and the remainder is 77.6000000 USDC"),
    ],
)
def test_chat_calculations_are_grounded_and_exact(prompt, expected):
    result = build_chat_stellar_calculation_context(prompt)
    assert result is not None
    assert expected in result
    assert "no transaction was made" in result


def test_non_calculation_stellar_question_has_no_calculation_context():
    assert build_chat_stellar_calculation_context("What is a Stellar trustline?") is None
