"""Adhkar corpus — offline, versioned supplication dataset with provenance.

Data source
-----------
``data/adhkar.json`` is a curated, hand-verified collection of supplications
drawn from Hisnul Muslim (Sa'id ibn 'Ali ibn Wahf al-Qahtani) and the two
Sahih collections.  Only entries graded *authentic* (sahih) or *good* (hasan)
by mainstream hadith scholarship are included.

Provenance contract
-------------------
Every ``AdhkarCorpus`` instance exposes:

* ``data_version`` — the semantic version string stored in the data file
  (``"version"`` key).  Bump this whenever entries are added, removed, or
  corrected.
* ``schema_version`` — integer schema revision; callers can branch on this if
  the shape of an entry ever changes.
* ``source`` — human-readable description of the upstream sources.
* ``data_last_updated`` — ISO-8601 date string from the data file indicating
  when the content was last reviewed.
* ``loaded_at`` — UTC ISO-8601 timestamp captured the moment the file was read
  into memory, surfaced in every API response so that operators can observe
  whether a stale in-process copy is being served after a hot reload.

Freshness note
--------------
The corpus is loaded once at import time (module-level ``corpus`` singleton).
If the data file is updated on disk, the process must be restarted for the
change to take effect.  The ``loaded_at`` field in the API response makes this
observable without log access.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DATA_PATH = Path(__file__).parent / "data" / "adhkar.json"


class AdhkarEntry(dict):
    """Simple typed container for a single adhkar/dua entry."""


class AdhkarCorpus:
    """Loads and searches the bundled adhkar dataset.

    Attributes
    ----------
    data_version:
        Semantic version of the data file (``"version"`` key).  ``"unknown"``
        when the key is absent.
    schema_version:
        Integer schema revision (``"schema_version"`` key).  ``0`` when absent.
    source:
        Human-readable upstream source description.
    data_last_updated:
        ISO-8601 date the corpus content was last reviewed/changed.
        ``"unknown"`` when the key is absent.
    loaded_at:
        UTC ISO-8601 timestamp at which this corpus instance loaded the data
        file.  Useful for detecting stale in-memory copies.
    entries:
        All parsed entries.
    """

    def __init__(self, data_file: Path = DATA_PATH) -> None:
        self.data_file = data_file
        self.entries: list[dict[str, Any]] = []
        self.data_version: str = "unknown"
        self.schema_version: int = 0
        self.source: str = "unknown"
        self.data_last_updated: str = "unknown"
        self.loaded_at: str = ""
        self._load_corpus()

    def _load_corpus(self) -> None:
        self.loaded_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if not self.data_file.exists():
            self.entries = []
            return
        with open(self.data_file, encoding="utf-8") as handle:
            payload = json.load(handle)
        self.entries = payload.get("entries", [])
        self.data_version = str(payload.get("version", "unknown"))
        self.schema_version = int(payload.get("schema_version", 0))
        self.source = str(payload.get("source_description", "unknown"))
        self.data_last_updated = str(payload.get("last_updated", "unknown"))

    def provenance(self) -> dict[str, Any]:
        """Return a provenance snapshot for inclusion in API responses."""
        return {
            "data_version": self.data_version,
            "schema_version": self.schema_version,
            "source": self.source,
            "data_last_updated": self.data_last_updated,
            "loaded_at": self.loaded_at,
            "entry_count": len(self.entries),
        }

    def search(self, *, category: str | None = None, query: str | None = None) -> list[dict[str, Any]]:
        """Return entries matching *category* and/or *query*.

        Both filters are optional; omitting both returns every entry.
        Matching is case-insensitive.  *query* is searched across the
        ``arabic``, ``transliteration``, ``translation``, ``category``, and
        ``source`` fields.
        """
        matches: list[dict[str, Any]] = []
        normalized_query = (query or "").strip().casefold()
        for entry in self.entries:
            category_matches = True
            if category:
                category_matches = str(entry.get("category", "")).casefold() == category.casefold()
            text_matches = True
            if normalized_query:
                haystack = " ".join(
                    [
                        str(entry.get("arabic", "")),
                        str(entry.get("transliteration", "")),
                        str(entry.get("translation", "")),
                        str(entry.get("category", "")),
                        str(entry.get("source", "")),
                    ]
                ).casefold()
                text_matches = normalized_query in haystack
            if category_matches and text_matches:
                matches.append(entry)
        return matches


corpus = AdhkarCorpus()
