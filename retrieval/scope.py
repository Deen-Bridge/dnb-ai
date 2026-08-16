"""Retrieval authorization scoping — deny-by-default.

Every retrieval call in the /chat handler must pass through a RetrievalScope
derived from the *authenticated* request context (API key + principal resolved
by the auth layer).  A scope is never built from the caller-supplied body
field ``ChatRequest.user_id``; that field is compared *against* the scope and
rejected with 403 if it disagrees.

Design goals
------------
* **Deny-by-default** — a missing or anonymous scope cannot access per-user
  data.  Only public content is accessible without a principal.
* **Single enforcement point** — every retriever (memory, purchase, semantic
  cache) accepts a ``RetrievalScope`` and delegates the allow/deny decision
  here, so the logic cannot silently diverge between call sites.
* **Post-check on every returned record** — ``visible_to`` must be called on
  every record returned from storage, not only on the lookup key.  Drop and
  log anything that slips through a fuzzy key match.
* **No publish-state leakage** — draft / private content is structurally
  unreachable: the scope's ``publish_visibility`` is ``"public"`` only for the
  public content paths; per-user paths default to ``"private"``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class UserRole(str, Enum):
    """Role carried by the principal. Extend as the auth layer grows."""

    ANONYMOUS = "anonymous"
    USER = "user"
    REVIEWER = "reviewer"
    ADMIN = "admin"


class PublishVisibility(str, Enum):
    """Content visibility expected from the retrieval path.

    ``PUBLIC``  — only published / public records are accessible.
    ``PRIVATE`` — per-principal records (the principal's own data only).
    """

    PUBLIC = "public"
    PRIVATE = "private"


# ---------------------------------------------------------------------------
# RetrievalScope
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetrievalScope:
    """Immutable scope token derived from the authenticated request context.

    Parameters
    ----------
    principal_user_id:
        The user id resolved from the auth layer (e.g. from the verified API
        key + session token or from the request's verified identity).  ``None``
        means the caller is anonymous; per-user data is structurally
        inaccessible.
    role:
        The principal's role.  Defaults to ``ANONYMOUS`` when no user is
        authenticated.
    publish_visibility:
        The expected visibility of content retrieved through this scope.
        Per-user memory paths always use ``PRIVATE``; public index paths use
        ``PUBLIC``.  A ``PUBLIC`` scope cannot access private records.
    """

    principal_user_id: str | None = None
    role: UserRole = UserRole.ANONYMOUS
    publish_visibility: PublishVisibility = PublishVisibility.PUBLIC

    # --- convenience factories ---------------------------------------------

    @classmethod
    def anonymous(cls) -> "RetrievalScope":
        """Unauthenticated caller — only public content accessible."""
        return cls(
            principal_user_id=None,
            role=UserRole.ANONYMOUS,
            publish_visibility=PublishVisibility.PUBLIC,
        )

    @classmethod
    def for_user(cls, user_id: str, role: UserRole = UserRole.USER) -> "RetrievalScope":
        """Authenticated user — grants access to that user's private content."""
        if not user_id or not user_id.strip():
            raise ValueError("user_id must be a non-empty string")
        return cls(
            principal_user_id=user_id,
            role=role,
            publish_visibility=PublishVisibility.PRIVATE,
        )


# ---------------------------------------------------------------------------
# Scope derivation from request context
# ---------------------------------------------------------------------------


def derive_scope(
    *,
    principal_user_id: str | None,
    role: UserRole = UserRole.ANONYMOUS,
) -> RetrievalScope:
    """Build a ``RetrievalScope`` from the authenticated request context.

    Call this *once* per request, at the top of the handler, using only
    values that come from a verified auth layer — never from the request body.

    Parameters
    ----------
    principal_user_id:
        User id resolved from the auth layer.  Pass ``None`` for unauthenticated
        (anonymous) traffic.
    role:
        Principal role.  Falls back to ``ANONYMOUS`` when ``principal_user_id``
        is ``None``.
    """
    if not principal_user_id:
        return RetrievalScope.anonymous()
    return RetrievalScope.for_user(principal_user_id, role)


# ---------------------------------------------------------------------------
# Cross-principal enforcement
# ---------------------------------------------------------------------------


def assert_scope_match(scope: RetrievalScope, requested_user_id: str | None) -> None:
    """Raise ``PermissionError`` when *requested_user_id* disagrees with the scope.

    This is the body-vs-principal check.  When the caller's request body
    contains a ``user_id`` that differs from the authenticated principal, the
    request must be rejected with 403 before any retrieval happens.

    Parameters
    ----------
    scope:
        The ``RetrievalScope`` derived from the authenticated context.
    requested_user_id:
        The ``user_id`` extracted from the request body (``ChatRequest.user_id``).

    Raises
    ------
    PermissionError
        When ``requested_user_id`` is set and does not match
        ``scope.principal_user_id``.
    """
    if requested_user_id is None:
        # Anonymous body field — no conflict.
        return
    if scope.principal_user_id is None:
        # Unauthenticated scope with a body user_id — deny.
        logger.warning(
            "Scope mismatch: unauthenticated request supplied user_id=%s; denied",
            requested_user_id[:8] if len(requested_user_id) >= 8 else requested_user_id,
        )
        raise PermissionError(
            "Cross-user access denied: unauthenticated request cannot supply user_id"
        )
    if scope.principal_user_id != requested_user_id:
        logger.warning(
            "Scope mismatch: principal=%s requested=%s; denied",
            scope.principal_user_id[:8],
            requested_user_id[:8] if len(requested_user_id) >= 8 else requested_user_id,
        )
        raise PermissionError(
            "Cross-user access denied: body user_id does not match authenticated principal"
        )


# ---------------------------------------------------------------------------
# Record-level post-check
# ---------------------------------------------------------------------------


def visible_to(record: Any, scope: RetrievalScope, *, owner_id_attr: str = "user_id") -> bool:
    """Post-retrieval ownership check — deny-by-default.

    Returns ``True`` when *record* is accessible under *scope*, ``False``
    otherwise.  A ``False`` result must cause the caller to drop the record
    and log the denial.

    Rules
    -----
    * A ``PUBLIC`` scope: the record must not carry a per-user owner attribute
      (or its owner attribute must be ``None``).
    * A ``PRIVATE`` scope with a principal: the record's owner must match
      ``scope.principal_user_id`` exactly.
    * Anonymous scope (principal is ``None``): always deny per-user records.

    Parameters
    ----------
    record:
        Any object.  The ``owner_id_attr`` attribute (default ``"user_id"``) is
        read to determine ownership.  If the attribute is absent the record is
        treated as public (no owner claim).
    scope:
        The ``RetrievalScope`` to check against.
    owner_id_attr:
        The attribute name that carries the owner's user id.
    """
    owner_id: str | None = getattr(record, owner_id_attr, None)

    if owner_id is None:
        # No ownership claim — public record; accessible to all scopes.
        return True

    # Record has an owner claim.
    if scope.principal_user_id is None:
        _log_denial(scope, owner_id, "anonymous scope cannot access owned record")
        return False

    if scope.principal_user_id != owner_id:
        _log_denial(scope, owner_id, "owner mismatch")
        return False

    if scope.publish_visibility == PublishVisibility.PUBLIC:
        # A PUBLIC scope must not serve private owned records.
        _log_denial(scope, owner_id, "public scope blocked per-user record")
        return False

    return True


def _log_denial(scope: RetrievalScope, owner_id: str, reason: str) -> None:
    principal_prefix = (
        scope.principal_user_id[:8] if scope.principal_user_id and len(scope.principal_user_id) >= 8
        else scope.principal_user_id or "anonymous"
    )
    owner_prefix = owner_id[:8] if len(owner_id) >= 8 else owner_id
    logger.warning(
        "retrieval_denied principal=%s owner=%s reason=%s",
        principal_prefix,
        owner_prefix,
        reason,
    )


# ---------------------------------------------------------------------------
# Scope cache-key helper
# ---------------------------------------------------------------------------


def cache_scope_key(scope: RetrievalScope, text_key: str) -> str:
    """Return a cache partition key that embeds the scope's principal.

    Public entries use the bare text key.
    Per-user entries are namespaced under ``user:{user_id}:``.

    This keeps a per-user cache entry structurally unreachable from any other
    user's lookup, even if two queries produce the same embedding vector.

    Parameters
    ----------
    scope:
        The ``RetrievalScope`` for the current request.
    text_key:
        The base cache key (e.g. normalized prompt or embedding hash).
    """
    if scope.principal_user_id is None:
        return text_key
    return f"user:{scope.principal_user_id}:{text_key}"
