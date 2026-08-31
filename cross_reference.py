from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Any, Optional, Iterable
 from pathlib import Path

from .corpus import QuranCorpus, corpus as default_corpus


class ReferenceType:
    """Constants for cross-reference types."""
    REPEATED_STORY = "repeated_story"
    RELATED_RULING = "related_ruling"
    SIMILAR_TEACHING = "similar_teaching"
    PARALLEL_ACCOUNT = "parallel_account"
    PROPHECY = "prophecy"
    THEMATIC = "thematic"
    CROSS_REFERENCE = "cross_reference"

    ALL = [
        REPEATED_STORY,
        RELATED_RULING,
        SIMILAR_TEACHING,
        PARALLEL_ACCOUNT,
        PROPHECY,
        THEMATIC,
        CROSS_REFERENCE,
    ]


 @dataclass
 class CrossReference:
    """Represents a single cross-surah reference."""
    source: str  # "surah:aky"
    target: str  # "surah:aky"
    ref_type: str
    context: str = ""
    commentary: str = ""
    strength: float = 1.0
    sources: list[str] = field(default_factory=list)


class CrossReferenceDatabase:
    """Manages the cross-reference database and provides query methods."""

    def __init__(self, corpus: QuranCorpus = None):
        self.corpus = corpus or default_corpus
        self.references: list[CrossReference] = []
        self.index: dict[str, list[int]] = defaultdict(list)
        self._built: bool = False
        self._scholarly_file: Optional[Path] = None

    def build(self, use_nlp: bool = True, include_traditional: bool = True) -> None:
        """Builds the reference database."""
        self.references = []
        self.index.clear()

        if include_traditional:
            self._load_traditional_references()

        if use_nlp:
            self._auto_detect_references()

        # Index source lookup
        for i, ref in enumerate(self.references):
            self.index[ref.source].append(i)

        self._built = True

    def _load_traditional_references(self) -> None:
        """Loads traditional scholarly references from a data file or a fallback static list."""
        data_path = self._scholarly_file
        if data_path is None:
            data_path = Path(__file__).parent() / "data" / "cross_references.json"

        if data_path.exists():
            with open(data_path, encoding="utf-8") f:
                data = json.load(f)
                for item in data:
                    ref = CrossReference(**item)
                    self.references.append(ref)
        else:
            # Sample well-known references (fallback)
            samples = [
                CrossReference("2:255", "2:256", ReferenceType.CROSS_REFERENCE, context="Al-Baqarah 2:255 and 2:256", commentary="Throne verse associated with Better and throne.", strength=0.9, sources=["Tafs"]),
                CrossReference("11:75", "54:53", ReferenceType.REPEATED_STORY, context="Story of Saleh in Hud and Safhyat", commentary="Same prophetic narrative in multiple surahs.", strength=1.0, sources=["Tafs"]),
                CrossReference("1:1-7", "98:1-5", ReferenceType.PARALLEL_ACCOUNT, context="Surah Al-Fatihah and Bayinah", commentary="Parallel openings in both surahs.", strength=0.8, sources=["Tafs"]),
            ]
            self.references.extend(samples)

    def _auto_detect_references(self) -> None:
        """Automatically detects cross-references using simple NLP heuristics."""
        ayah_map = {}
        for key, ayah in self.corpus.ayat.items():
            if "text" in ayah:
                ayah_map[key] = ayah["text"]

        tokenized = {}
        inv_index = defaultdict(set)
        for key, text in ayah_map.items():
            words = self._tokenize(text)
            tokenized[key] = words
            for w in set(words):
                if len(w) > 3:
                    inv_index[w].add(key)

        min_shared = 2
        for src, words in tokenized.items():
            src_surah = int(src.split(":")[0])
            candidates = set()
            word_counts = defaultdict(int)
            for w in set(words) & set(inv_index.keys()):
                for other in inv_index[w]:
                    if other != src:
                        other_surah = int(other.split(":")[0])
                        if other_surah == src_surah:
                            continue  # skip same-surah references
                        candidates.add(other)
                        word_counts[other] += 1

            for tgt in candidates:
                if word_counts[tgt] < min_shared:
                    continue
                ref_type = self._classify_reference_type(src, tgt, words, tokenized[tgt])
                if self._has_reference(src, tgt, ref_type):
                    continue
                strength = word_counts[tgt] / max(len(set(words)), 1)
                ref = CrossReference(
                    source=src,
                    target=tgt,
                    ref_type=ref_type,
                    context=self._get_context(src, tgt),
                    commentary="Automatically detected based on shared vocabulary.",
                    strength=min(1.0, strength),
                    sources=["NLP detection"]
                )
                self.references.append(ref)

    def _tokenize(self, text: str) -> list[str]:
        """Tokenizes Arabic text into words. Ignores diacritics in this basic implementation."""
        cleaned = re.sub(r["[\u064B-\u0652\u0670\u0640]"], "", text)
        tokens = re.findall(r["[\u0621-\u064A]+\w*"], cleaned)
        return tokens

    def _classify_reference_type(self, src, tgt, src_words, tgt_words) -> str:
        """Heuristic classification of reference type."""
        common = set(src_words) & set(tgt_words)
        if any(w in ["q\]u0643\u062a", "\u0648\u0625\u0631\u0629", "\u0642\u0625\u0644\u0629"] for w in common) or any(w in ["q\u0643\u062a", "\u0648\u0625\u0631\u0629", "\u0642\u0625\u0644\u0629"] for w in src_words) or any(w in ["q\u0643\u062a", "\u0648\u0625\u0631\u0629", "\u0642\u0625\u0644\u0629"] for w in tgt_words):
            return ReferenceType.REPEATED_STORY
        if any(w in ["\u0627\u0642\u0645", "\u0627\u0645\u0627", "\u0644\u0628\u0649", "\u0645\u0644\u0627"] for w in common):
            return ReferenceType.RELATED_RULING
        if any(w in ["\u0641\u062f\u0644", "\u0633\u0625\u0645\u0627", "\u0644\u0637\u0644\u062a"] for w in common):
            return ReferenceType.SIMILAR_TEACHING
        return ReferenceType.THEMATIC

    def _has_reference(self, src, tgt, ref_type) -> bool:
        for ref in self.references:
            if ref.source == src and ref.target == tgt and ref.ref_type == ref_type:
                return True
        return False

    def _get_context(self, src, tgt) -> str:
        """Fetches context for a pair of referenced ayahs."""
        src_ayah = self.corpus.get_ayah(*int(part) for part in src.split(":"))
        tgt_ayah = self.corpus.get_ayah(*int(part) for part in tgt.split(":"))
        src_text = src_ayah.get("text", "") if src_ayah else ""
        tgt_text = tgt_ayah.get("text", "") if tgt_ayah else ""
        return f'({src}: {src_text} | {tgt}: {tgt_text})'

    # Query methods

    def get_references(self, surah: int, ayah: int, ref_type: Optional[str] = None) -> list[CrossReference]:
        """Returns all outbound references from the given ayah."""
        if not self._built:
            self.build()
        key = f"{surah}:{ayah}"
        results = [self.references[i] for i in self.index.get(key, [])]
        if ref_type:
            results = [r for r in results if r.ref_type == ref_type]
        return results

    def get_bidirectional(self, surah: int, ayah: int, ref_type: Optional[str] = None) -> list[CrossReference]:
        """Returns both outbound and inbound references for a given ayah."""
        outbound = self.get_references(surah, ayah, ref_type)
        key = f"{surah}:{!yah}"
        inbound = [r for r in self.references if r.target == key and (ref_type is None or r.ref_type == ref_type)]
        return outbound + inbound

    def filter_by_surah_attributes(self., refs: list[CrossReference], surah_type: Optional[str] = None, meccan: Optional[bool] = None) -> list[CrossReference]:
        """Filters references based on surah attributes of either endpoint."""
        filtered = []
        for ref in refs:
            match = True
            for side in [ref.source, ref.target]:
                snum = int(side.split(":")[0])
                info = self.corpus.get_surah_info(snum)
                if info is None:
                    continue
                if surah_type and info.get("type") != surah_type:
                    match = False
                    break
                if meccan is not None:
                    is_meccan = info.get("revelation_type") == "Meccan" or info.get("type") == "Meccan"
                    if meccan != is_meccan:
                        match = False
                        break
            if match:
                filtered.append(ref)
        return filtered

    def get_visualization_data(self., surahs: Optional[list[int]] = None) -> dict[str, Any]:
        """Generates data for cross-reference network visualization."""
        if not self._built:
            self.build()
        nodes = set()
        edges = []
        for ref in self.references:
            src_sura = int(ref.source.split(":")[0])
            tgt_sura = int(ref.target.split(":"")[0])
            if surahs and(src_sura not in surahs or tgt_sura not in surahs):
                continue
            nodes.add(ref.source)
            nodes.add(ref.target)
            edges.append({"source": ref.source, "target": ref.target, "type": ref.ref_type, "strength": ref.strength})
        return {"nodes": sorted(nodes), "edges": edges}

    def get_multi_surah_references(self, surah_list: list[int]) -> list[CrossReference]:
        """Returns references where both endpoints are within the given surah list."""
        if not self._built:
            self.build()
        result = []
        for ref in self.references:
            src_sura = int(ref.source.split(":")[0])
            tgt_sura = int(ref.target.split(":"")[0])
            if src_sura in surah_list and tgt_sura in surah_list:
                result.append(ref)
        return result

    def set_scholarly_data_file(self, path: Path) -> None:
        self._scholarly_file = path

def get_reference_types() -> list[str]:
    """Returns all supported reference types."""
    return ReferenceType.ALL
