"""Small, deterministic public-knowledge catalog for discovery queries.

Only explicitly published records are exposed; callers cannot retrieve draft or
private fields through this module. A later indexer can replace the in-memory
source without changing the filtering contract.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class PublicRecord:
    kind: str
    title: str
    summary: str
    url: str
    published: bool = True


class PublicKnowledgeCatalog:
    def __init__(self, records: list[PublicRecord] | None = None) -> None:
        self._records = tuple(records or ())

    def search(self, query: str, *, kind: str | None = None) -> list[PublicRecord]:
        terms = {part.casefold() for part in query.split() if part.strip()}
        if not terms:
            return []
        matches: list[PublicRecord] = []
        for record in self._records:
            if not record.published or (kind and record.kind != kind):
                continue
            haystack = f"{record.title} {record.summary}".casefold()
            if all(term in haystack for term in terms):
                matches.append(record)
        return matches
