"""Tests for OpenAPI tags metadata, examples, and error response models (#18)."""

from __future__ import annotations

from openapi_metadata import OpenAPIErrorModel, get_openapi_tags_and_error_responses


def test_openapi_tags_and_standard_error_examples_are_complete() -> None:
    tags, errors = get_openapi_tags_and_error_responses()
    tag_names = {t["name"] for t in tags}
    assert {"chat", "languages", "stellar", "health"}.issubset(tag_names)
    assert {400, 429, 503}.issubset(errors.keys())
    err = OpenAPIErrorModel(code="invalid_request", message="Bad input", retryable=False)
    assert err.retryable is False
