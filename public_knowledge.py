"""Governed public-knowledge catalog for discovery queries (#848).

Only records that are explicitly published *and* carry complete provenance are
searchable. Everything else fails closed:

* an unpublished record is dropped at construction and never kept, so no
  accessor can return its draft or private fields;
* a published record missing provenance, with an unsafe URL, or a duplicate of
  an earlier record is excluded from search and reported in ``rejected`` by
  kind, URL and reason only (its title and summary are withheld).

Each searchable record states where it came from (``source``), the rights basis
for publishing it (``license``), the version of the source data it was built
from (``dataset_version``), who approved it (``reviewed_by``, a role rather
than a person) and when (``last_reviewed``), plus an optional ``caveat`` for
uncertainty or scope. Search results are those same records, so every output
carries its provenance.

The catalog holds public records only. It has no user identifiers and must
never be given user-scoped data; learner state lives in ``study_store`` and is
governed separately. Search is deterministic: records keep insertion order and
matching is case-insensitive, ignores Latin transliteration marks ("Ṣaḥīḥ"
matches "sahih", "Qurʾān" matches "Quran"), and is Arabic-aware: tashkeel,
tatweel, presentation forms and alef/ya/ta-marbuta variants are folded, the
last through ``quran_recitation.normalize_arabic``.

Who may publish, how sources are versioned, the review cadence and rollback are
in ``docs/PUBLIC_KNOWLEDGE_GOVERNANCE.md``.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, timedelta

from quran_recitation import normalize_arabic

CATALOG_SCHEMA_VERSION = "2"

MAX_QUERY_CHARS = 256
MAX_QUERY_TERMS = 16

# Relative site paths ("/events/1") or absolute https URLs only. Anything else
# (javascript:, data:, http:, protocol-relative "//host") is rejected.
_SAFE_URL = re.compile(r"^(?:/(?![/\\])|https://[^\s/?#]+)[^\s<>\"'`]*$")

# A version label such as "1", "1.0.0" or "2026-08-30": no spaces or markup.
_VERSION_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,63}$")

_REQUIRED_TEXT_FIELDS = ("kind", "title", "source", "license", "reviewed_by")

# Apostrophe-like marks used in transliteration (Qur'an, Qurʾān, ʿIlm). Removed,
# not spaced, so every spelling of a word folds to the same token.
_TRANSLITERATION_MARKS = re.compile("[\u02bc\u02be\u02bf\u2018\u2019'`]")


@dataclass(frozen=True)
class PublicRecord:
    """One public catalog entry. The first five fields keep their original order."""

    kind: str
    title: str
    summary: str
    url: str
    published: bool = True
    source: str = ""
    """Canonical source the record is drawn from, e.g. a dataset path or a publisher."""
    license: str = ""
    """Rights basis for publishing it, e.g. "CC BY 4.0" or "Public domain"."""
    dataset_version: str = ""
    """Version of the source data the record was built from, e.g. "1.0.0"."""
    reviewed_by: str = ""
    """Role or team that approved publication. Never a personal name or email."""
    last_reviewed: str = ""
    """ISO date (YYYY-MM-DD) of the most recent review."""
    caveat: str = ""
    """Uncertainty or scope note shown with the record; empty when there is none."""


@dataclass(frozen=True)
class RejectedRecord:
    """A published record that failed validation. Title and summary are withheld."""

    kind: str
    url: str
    reasons: tuple[str, ...]


def _parse_review_date(value: str) -> date | None:
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def validate_record(record: PublicRecord, *, today: date | None = None) -> list[str]:
    """Reasons a published record cannot be served. Empty means it is valid."""
    reasons: list[str] = []
    for name in _REQUIRED_TEXT_FIELDS:
        if not str(getattr(record, name)).strip():
            reasons.append(f"missing {name}")
    if not _SAFE_URL.match(record.url or ""):
        reasons.append("url must be a site path starting with '/' or an https:// URL")
    if not _VERSION_LABEL.match(record.dataset_version or ""):
        reasons.append("missing or malformed dataset_version")
    reviewed = _parse_review_date(record.last_reviewed or "")
    if reviewed is None:
        reasons.append("last_reviewed must be an ISO date (YYYY-MM-DD)")
    elif reviewed > (today or date.today()):
        reasons.append("last_reviewed is in the future")
    return reasons


def normalize_search_text(text: str) -> str:
    """Fold text for matching.

    NFKD decomposition expands presentation forms and ligatures, and dropping
    the resulting combining marks removes both Latin diacritics and Arabic
    tashkeel. Transliteration apostrophes are removed, then
    ``normalize_arabic`` folds alef/ya/ta-marbuta variants and turns
    punctuation into spaces, and the result is casefolded.
    """
    decomposed = unicodedata.normalize("NFKD", text or "")
    bare = "".join(char for char in decomposed if not unicodedata.combining(char))
    return normalize_arabic(_TRANSLITERATION_MARKS.sub("", bare)).casefold()


class PublicKnowledgeCatalog:
    """Read-only, versioned catalog of published, provenance-complete records."""

    def __init__(
        self,
        records: Iterable[PublicRecord] | None = None,
        *,
        version: str = "unversioned",
        today: date | None = None,
    ) -> None:
        self.version = version
        accepted: list[PublicRecord] = []
        rejected: list[RejectedRecord] = []
        seen: set[tuple[str, str]] = set()
        unpublished = 0

        for record in records or ():
            if not record.published:
                unpublished += 1
                continue
            reasons = validate_record(record, today=today)
            identity = (record.kind, record.url)
            if not reasons and identity in seen:
                reasons = ["duplicate of an earlier published record with the same kind and url"]
            if reasons:
                rejected.append(RejectedRecord(kind=record.kind, url=record.url, reasons=tuple(reasons)))
                continue
            seen.add(identity)
            accepted.append(record)

        self._records = tuple(accepted)
        self._haystacks = tuple(normalize_search_text(f"{r.title} {r.summary}") for r in accepted)
        self._rejected = tuple(rejected)
        self._unpublished = unpublished

    @property
    def records(self) -> tuple[PublicRecord, ...]:
        """Every searchable record, in insertion order."""
        return self._records

    @property
    def rejected(self) -> tuple[RejectedRecord, ...]:
        """Published records excluded for failing validation, with reasons."""
        return self._rejected

    @property
    def unpublished_count(self) -> int:
        """How many unpublished records were dropped. Their content is not kept."""
        return self._unpublished

    def search(self, query: str, *, kind: str | None = None) -> list[PublicRecord]:
        """Records whose title or summary contains every query term.

        An empty or whitespace-only query returns nothing, so an empty search
        never lists the catalog. A query longer than MAX_QUERY_CHARS or with
        more than MAX_QUERY_TERMS terms raises ValueError.
        """
        if len(query) > MAX_QUERY_CHARS:
            raise ValueError(f"query exceeds {MAX_QUERY_CHARS} characters")
        terms = list(dict.fromkeys(normalize_search_text(query).split()))
        if not terms:
            return []
        if len(terms) > MAX_QUERY_TERMS:
            raise ValueError(f"query has more than {MAX_QUERY_TERMS} terms")

        matches: list[PublicRecord] = []
        for record, haystack in zip(self._records, self._haystacks, strict=True):
            if kind and record.kind != kind:
                continue
            if all(term in haystack for term in terms):
                matches.append(record)
        return matches

    def needs_review(self, *, as_of: date, max_age_days: int) -> list[PublicRecord]:
        """Searchable records whose last review is older than ``max_age_days``.

        Stale records stay searchable; this list is for the review queue, not
        for hiding content.
        """
        if max_age_days < 0:
            raise ValueError("max_age_days must not be negative")
        cutoff = as_of - timedelta(days=max_age_days)
        stale: list[PublicRecord] = []
        for record in self._records:
            reviewed = _parse_review_date(record.last_reviewed)
            if reviewed is not None and reviewed < cutoff:
                stale.append(record)
        return stale
