import json
from pathlib import Path
from typing import Any

DATA_PATH = Path(__file__).parent / "data" / "quran_uthmani.json"
RELATIONSHIP_DATA_PATH = Path(__file__).parent / "data" / "quran_relationships.json"


class QuranCorpus:
    def __init__(self, data_file: Path = DATA_PATH, relationship_file: Path = RELATIONSHIP_DATA_PATH):
        self.data_file = data_file
        self.relationship_file = relationship_file
        self.surahs: dict[str, dict[str, Any]] = {}
        self.ayat: dict[str, dict[str, str]] = {}
        self.relationships: dict[str, list[dict[str, Any]]] = {}
        self.scholarly_notes: dict[str, list[str]] = {}
        self._load_corpus()
        self._load_relationships()

    def _load_corpus(self) -> None:
        if self.data_file.exists():
            with open(self.data_file, encoding="utf-8") as f:
                content = json.load(f)
                self.surahs = content.get("surahs", {})
                self.ayat = content.get("ayat", {})
        else:
            self.surahs = {}
            self.ayat = {}

    def _load_relationships(self) -> None:
        if self.relationship_file.exists():
            with open(self.relationship_file, encoding="utf-8") as f:
                self.relationships = json.load(f)

    def _get_ayah_text(self, surah: int, ayah: int) -> str:
        ayah_data = self.get_ayah(surah, ayah)
        return ayah_data.get("text", "") if ayah_data else ""

    def _tokenize(self, text: str) -> set[str]:
        return {w for w in text.split() if len(w) > 1}

    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        set1 = self._tokenize(text1)
        set2 = self._tokenize(text2)
        if not set1 or not set2:
            return 0.0
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union else 0.0

    def _classify_relationship(self, text1: str, text2: str, score: float) -> str:
        # Simple classification based on shared thematic keywords.
        keywords = ["salat", "zakat", "sawm", "rahmah", "iman", "kufr", "jihad"]
        if any(w in self._tokenize(text1) & self._tokenize(text2) for w in keywords):
            return "parallel"
        return "elaboration"

    def build_relationships(self, threshold: float = 0.2) -> None:
        ayat_keys = list(self.ayat.keys())
        for i in range(len(ayat_keys)):
            key1 = ayat_keys[i]
            for j in range(i + 1, len(ayat_keys)):
                key2 = ayat_keys[j]
                text1 = self._get_ayah_text(*map(int, key1.split(":")))
                text2 = self._get_ayah_text(*map(int, key2.split(":")))
                score = self._jaccard_similarity(text1, text2)
                if score >= threshold:
                    rel_type = self._classify_relationship(text1, text2, score)
                    entry1 = {"target": key2, "type": rel_type, "strength": score}
                    entry2 = {"target": key1, "type": rel_type, "strength": score}
                    self.relationships.setdefault(key1, []).append(entry1)
                    self.relationships.setdefault(key2, []).append(entry2)
        self._save_relationships()

    def _save_relationships(self) -> None:
        self.relationship_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.relationship_file, "w", encoding="utf-8") as f:
            json.dump(self.relationships, f, ensure_ascii=True)

    def get_related_ayah(
        self,
        surah: int,
        ayah: int,
        relationship_type: str | None = None,
        min_strength: float = 0.0,
        max_depth: int = 1,
    ) -> list[dict[str, Any]]:
        key = f"{surah}:{ayah}"
        if key not in self.relationships:
            return []
        direct = self.relationships[key]
        if max_depth == 1:
            results = []
            for rel in direct:
                if (relationship_type is None or rel["type"] == relationship_type) and rel["strength"] >= min_strength:
                    copy = rel.copy()
                    copy["source"] = key
                    results.append(copy)
            return results

        # Breadth-first search for indirect relationships.
        visited = set()
        queue = [(key, 0)]
        all_results = []
        while queue:
            curr_key, depth = queue.pop(0)
            if depth > max_depth or curr_key in visited:
                continue
            visited.add(curr_key)
            if curr_key != key:
                for rr in self.relationships.get(curr_key, []):
                    if (relationship_type is None or rr["type"] == relationship_type) and rr[
                        "strength"
                    ] >= min_strength:
                        copy = rr.copy()
                        copy["source"] = curr_key
                        copy["depth"] = depth
                        all_results.append(copy)
            for rr in self.relationships.get(curr_key, []):
                if rr["target"] not in visited:
                    queue.append((rr["target"], depth + 1))
        return all_results

    def get_relationship_graph(self, surah: int | None = None, ayah: int | None = None) -> Any:
        if surah is None and ayah is None:
            return self.relationships
        key = f"{surah}:{ayah}"
        return self.relationships.get(key, {})

    def add_scholarly_note(self, surah: int, ayah: int, note: str) -> None:
        key = f"{surah}:{ayah}"
        self.scholarly_notes.setdefault(key, []).append(note)

    def get_scholarly_notes(self, surah: int, ayah: int) -> list[str]:
        key = f"{surah}:{ayah}"
        return self.scholarly_notes.get(key, [])

    def get_surah_info(self, surah: int) -> dict[str, Any] | None:
        return self.surahs.get(str(surah))

    def get_ayah_count(self, surah: int) -> int | None:
        info = self.get_surah_info(surah)
        return info["ayahs_count"] if info else None

    def get_ayah(self, surah: int, ayah: int) -> dict[str, str] | None:
        key = f"{surah}:{ayah}"
        return self.ayat.get(key)

    def has_hadith_corpus(self) -> bool:
        # Stub accessor for compatibility with Issue #24
        return False


# Shared instance across the application
corpus = QuranCorpus()

if __name__ == "__main__":
    # Build relationships on demand if file missing.
    if not corpus.relationship_file.exists():
        print("Relationship data not found. Building now...")
        corpus.build_relationships(threshold=0.2)
        print(f"Relationships built and saved to {corpus.relationship_file}")
