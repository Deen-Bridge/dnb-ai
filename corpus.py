import json
from pathlib import Path
from typing import Any

DATA_PATH = Path(__file__).parent / "data" / "quran_uthmani.json"


class QuranCorpus:
    def __init__(self, data_file: Path = DATA_PATH):
        self.data_file = data_file
        self.surahs: dict[str, dict[str, Any]] = {}
        self.ayat: dict[str, dict[str, str]] = {}
        self._load_corpus()

    def _load_corpus(self) -> None:
        if self.data_file.exists():
            with open(self.data_file, encoding="utf-8") as f:
                content = json.load(f)
                self.surahs = content.get("surahs", {})
                self.ayat = content.get("ayat", {})
        else:
            self.surahs = {}
            self.ayat = {}

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
