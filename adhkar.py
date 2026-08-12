import json
from pathlib import Path
from typing import Any

DATA_PATH = Path(__file__).parent / "data" / "adhkar.json"


class AdhkarEntry(dict):
    """Simple typed container for a single adhkar/dua entry."""


class AdhkarCorpus:
    def __init__(self, data_file: Path = DATA_PATH):
        self.data_file = data_file
        self.entries: list[dict[str, Any]] = []
        self._load_corpus()

    def _load_corpus(self) -> None:
        if not self.data_file.exists():
            self.entries = []
            return
        with open(self.data_file, encoding="utf-8") as handle:
            payload = json.load(handle)
            self.entries = payload.get("entries", [])

    def search(self, *, category: str | None = None, query: str | None = None) -> list[dict[str, Any]]:
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
