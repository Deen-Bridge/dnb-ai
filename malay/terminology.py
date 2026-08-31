from __future__ import annotations

import json
import os
from typing import Any

class TerminologyLexicon:
    def __init__(self) -> None:
        self.terms: list[dict[str, Any]] = []
        self._load_lexicon()

    def _load_lexicon(self) -> None:
        path = os.path.join("data", "malay_indonesian_islamic_terms.json")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.terms = data.get("terms", [])
            except Exception:
                self.terms = []
        if not self.terms:
            # Fallback default terms
            self.terms = [
                {
                    "id": "ms-id-ibada-solat",
                    "malay_term": "Solat",
                    "indonesian_term": "Shalat",
                    "jawi_term": "صلاة",
                    "variants": ["solat", "shalat", "sembahyang", "salat"]
                }
            ]

    def lookup(self, term: str) -> dict[str, Any] | None:
        tl = term.lower()
        for t in self.terms:
            if tl == t.get("malay_term", "").lower() or tl == t.get("indonesian_term", "").lower():
                return t
            if tl in [v.lower() for v in t.get("variants", [])]:
                return t
        return None

    def identify_terms(self, text: str) -> list[dict[str, Any]]:
        found = []
        words = text.split()
        for w in words:
            cleaned = w.strip(",.?!;:()").lower()
            res = self.lookup(cleaned)
            if res and res not in found:
                found.append(res)
        return found
