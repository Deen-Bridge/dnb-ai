"""Coalesce duplicate concurrent read-only Stellar provider lookups (#576)."""

from __future__ import annotations

import copy
import threading
from collections.abc import Callable
from concurrent.futures import Future
from typing import Any


def build_public_lookup_key(*, network: str, resource_type: str, identifier: str) -> str:
    """Construct a privacy-safe public cache/coalescing key excluding user context."""
    net = network.strip().lower()
    rtype = resource_type.strip().lower()
    ident = identifier.strip()
    if not net or not rtype or not ident:
        raise ValueError("network, resource_type, and identifier must be non-empty")
    return f"{net}:{rtype}:{ident}"


class PublicLookupCoalescer:
    """Deduplicate in-flight read-only Stellar lookups by public key only."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._inflight: dict[str, Future[dict[str, Any]]] = {}

    def execute_or_join(
        self,
        *,
        network: str,
        resource_type: str,
        identifier: str,
        fetcher: Callable[[], dict[str, Any]],
    ) -> dict[str, Any]:
        """Coalesce concurrent calls sharing the same public lookup key."""
        key = build_public_lookup_key(network=network, resource_type=resource_type, identifier=identifier)
        with self._lock:
            existing = self._inflight.get(key)
            if existing is not None:
                is_leader = False
                fut = existing
            else:
                is_leader = True
                fut = Future()
                self._inflight[key] = fut

        if is_leader:
            try:
                result = fetcher()
                fut.set_result(result)
            except Exception as exc:
                fut.set_exception(exc)
            finally:
                with self._lock:
                    self._inflight.pop(key, None)

        return copy.deepcopy(fut.result())
