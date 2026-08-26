"""Narrator biography (rijal) lookup system — static, offline, JSON-backed.

Provides structured narrator profiles with reliability assessments, teacher-student
networks, and isnad chain resolution for hadith science research.
"""

from __future__ import annotations

import json
import os
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/narrators", tags=["narrators"])

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ReliabilityAssessment(BaseModel):
    scholar: str
    rating: str


class NarratorSummary(BaseModel):
    id: str
    name: str
    kunyah: str | None = None
    nisba: str | None = None
    laqab: str | None = None
    region: str | None = None
    relevance_score: float = Field(ge=0.0, le=1.0)


class NarratorProfile(BaseModel):
    id: str
    name: str
    kunyah: str | None = None
    nisba: str | None = None
    laqab: str | None = None
    birth_year: int | None = None
    death_year: int | None = None
    birth_year_hijri: int | None = None
    death_year_hijri: int | None = None
    region: str | None = None
    reliability_assessment: list[ReliabilityAssessment] = Field(default_factory=list)
    teachers: list[str] = Field(default_factory=list)
    students: list[str] = Field(default_factory=list)
    narrated_hadiths_count: int | None = None
    biography_summary: str | None = None


class ComparisonResult(BaseModel):
    narrators: list[NarratorProfile]
    shared_teachers: list[str] = Field(default_factory=list)
    shared_students: list[str] = Field(default_factory=list)


class NetworkNode(BaseModel):
    id: str
    name: str
    relation: str  # "self", "teacher", "student"


class NetworkGraph(BaseModel):
    center: str
    nodes: list[NetworkNode]
    depth: int


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------


def _default_data_path() -> str:
    env = os.getenv("NARRATOR_DB_PATH")
    if env:
        return env
    return str(Path(__file__).parent / "data" / "narrators.json")


class NarratorDatabase:
    def __init__(self, path: str | None = None) -> None:
        self._path = path or _default_data_path()
        self._records: list[dict[str, Any]] = []
        self._by_id: dict[str, dict[str, Any]] = {}
        self._loaded = False

    # -- lazy load --

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with open(self._path, encoding="utf-8") as fh:
            self._records = json.load(fh)
        self._by_id = {r["id"]: r for r in self._records}
        self._loaded = True

    # -- helpers --

    @staticmethod
    def _score(query: str, record: dict[str, Any]) -> float:
        """Return a 0-1 relevance score for a query against a narrator record."""
        q = query.lower().strip()
        fields: list[tuple[str, float]] = []
        if record.get("name"):
            fields.append((record["name"].lower(), 1.0))
        if record.get("kunyah"):
            fields.append((record["kunyah"].lower(), 0.9))
        if record.get("nisba"):
            fields.append((record["nisba"].lower(), 0.85))
        if record.get("laqab") and record["laqab"] != "none":
            fields.append((record["laqab"].lower(), 0.8))
        if record.get("id"):
            fields.append((record["id"].replace("-", " "), 0.7))

        best = 0.0
        for text, weight in fields:
            if q in text:
                best = max(best, weight)
            else:
                if len(q) <= 12 and len(q) <= len(text) * 2:
                    ratio = SequenceMatcher(None, q, text).ratio()
                    best = max(best, ratio * weight)
        return min(best, 1.0)

    @staticmethod
    def _to_summary(record: dict[str, Any], score: float) -> NarratorSummary:
        return NarratorSummary(
            id=record["id"],
            name=record["name"],
            kunyah=record.get("kunyah"),
            nisba=record.get("nisba"),
            laqab=record.get("laqab") if record.get("laqab") != "none" else None,
            region=record.get("region"),
            relevance_score=round(score, 3),
        )

    @staticmethod
    def _to_profile(record: dict[str, Any]) -> NarratorProfile:
        assessments = [
            ReliabilityAssessment(scholar=s, rating=r)
            for s, r in record.get("reliability_assessment", {}).items()
        ]
        return NarratorProfile(
            id=record["id"],
            name=record["name"],
            kunyah=record.get("kunyah"),
            nisba=record.get("nisba"),
            laqab=record.get("laqab") if record.get("laqab") != "none" else None,
            birth_year=record.get("birth_year"),
            death_year=record.get("death_year"),
            birth_year_hijri=record.get("birth_year_hijri"),
            death_year_hijri=record.get("death_year_hijri"),
            region=record.get("region"),
            reliability_assessment=assessments,
            teachers=record.get("teachers", []),
            students=record.get("students", []),
            narrated_hadiths_count=record.get("narrated_hadiths_count"),
            biography_summary=record.get("biography_summary"),
        )

    # -- public API --

    def search(
        self,
        query: str,
        *,
        min_score: float = 0.2,
        max_results: int = 20,
    ) -> list[NarratorSummary]:
        self._ensure_loaded()
        scored: list[tuple[float, dict[str, Any]]] = []
        for rec in self._records:
            s = self._score(query, rec)
            if s >= min_score:
                scored.append((s, rec))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [self._to_summary(r, s) for s, r in scored[:max_results]]

    def get_narrator(self, narrator_id: str) -> NarratorProfile:
        self._ensure_loaded()
        rec = self._by_id.get(narrator_id)
        if rec is None:
            raise HTTPException(status_code=404, detail=f"Narrator '{narrator_id}' not found")
        return self._to_profile(rec)

    def compare_narrators(self, ids: list[str]) -> ComparisonResult:
        profiles = [self.get_narrator(nid) for nid in ids]
        if len(profiles) < 2:
            return ComparisonResult(narrators=profiles)
        all_teachers = [set(p.teachers) for p in profiles]
        all_students = [set(p.students) for p in profiles]
        shared_t = sorted(set.intersection(*all_teachers)) if all_teachers else []
        shared_s = sorted(set.intersection(*all_students)) if all_students else []
        return ComparisonResult(narrators=profiles, shared_teachers=shared_t, shared_students=shared_s)

    def lookup_isnad(self, names: list[str]) -> list[NarratorProfile]:
        results: list[NarratorProfile] = []
        seen: set[str] = set()
        for name in names:
            matches = self.search(name, min_score=0.3, max_results=1)
            if matches and matches[0].id not in seen:
                results.append(self.get_narrator(matches[0].id))
                seen.add(matches[0].id)
        return results

    def get_network(self, narrator_id: str, depth: int = 1) -> NetworkGraph:
        self._ensure_loaded()
        rec = self._by_id.get(narrator_id)
        if rec is None:
            raise HTTPException(status_code=404, detail=f"Narrator '{narrator_id}' not found")
        nodes: list[NetworkNode] = [
            NetworkNode(id=rec["id"], name=rec["name"], relation="self")
        ]
        visited: set[str] = {rec["id"]}
        frontier = {rec["id"]}
        for _ in range(depth):
            next_frontier: set[str] = set()
            for fid in frontier:
                frec = self._by_id.get(fid)
                if frec is None:
                    continue
                for tid in frec.get("teachers", []):
                    if tid not in visited and tid in self._by_id:
                        visited.add(tid)
                        nodes.append(NetworkNode(id=tid, name=self._by_id[tid]["name"], relation="teacher"))
                        next_frontier.add(tid)
                for sid in frec.get("students", []):
                    if sid not in visited and sid in self._by_id:
                        visited.add(sid)
                        nodes.append(NetworkNode(id=sid, name=self._by_id[sid]["name"], relation="student"))
                        next_frontier.add(sid)
            frontier = next_frontier
        return NetworkGraph(center=narrator_id, nodes=nodes, depth=depth)

    @property
    def count(self) -> int:
        self._ensure_loaded()
        return len(self._records)


_db = NarratorDatabase()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/search")
def narrators_search(q: str, max_results: int = 20) -> list[NarratorSummary]:
    """Search narrators by name, kunyah, nisba, laqab, or id."""
    return _db.search(q, max_results=max_results)


@router.get("/{narrator_id}", response_model=NarratorProfile)
def get_narrator(narrator_id: str) -> NarratorProfile:
    """Get full biography and reliability assessment for a narrator."""
    return _db.get_narrator(narrator_id)


class CompareRequest(BaseModel):
    ids: list[str] = Field(..., min_length=2, max_length=10)


@router.post("/compare", response_model=ComparisonResult)
def compare_narrators(body: CompareRequest) -> ComparisonResult:
    """Compare two or more narrators side-by-side."""
    return _db.compare_narrators(body.ids)


class IsnadLookupRequest(BaseModel):
    names: list[str] = Field(..., min_length=1, max_length=50)


@router.post("/isnad-lookup", response_model=list[NarratorProfile])
def isnad_lookup(body: IsnadLookupRequest) -> list[NarratorProfile]:
    """Resolve a list of narrator names from an isnad chain."""
    return _db.lookup_isnad(body.names)


@router.get("/{narrator_id}/network", response_model=NetworkGraph)
def narrator_network(narrator_id: str, depth: int = 1) -> NetworkGraph:
    """Get the teacher-student network graph for a narrator."""
    return _db.get_network(narrator_id, depth=depth)
