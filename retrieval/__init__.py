"""Retrieval authorization layer — deny-by-default scoping.

Import ``RetrievalScope``, ``derive_scope``, and the enforcement helpers from
this package rather than from ``retrieval.scope`` directly so downstream code
has a stable import surface.
"""

from retrieval.scope import (
    PublishVisibility,
    RetrievalScope,
    UserRole,
    assert_scope_match,
    cache_scope_key,
    derive_scope,
    visible_to,
)

__all__ = [
    "PublishVisibility",
    "RetrievalScope",
    "UserRole",
    "assert_scope_match",
    "cache_scope_key",
    "derive_scope",
    "visible_to",
]
