
import json
from pathlib import Path
from typing import Dict, Any, Optional

DATA_PATH = Path(__file__).parent / "data" / "quran_uthmani.json"


class QuranCorpus:
    def __init__(self, data_file: Path = DATA_PATH):
        self.data_file = data_file
        self.surahs: Dict[str, Dict[str, Any]] = {}
        self.ayat: Dict[str, Dict[str, str]] = {}
        self._load_corpus()

    def _load_corpus(self) -> None:
        if self.data_file.exists():
            with open(self.data_file, "r", encoding="utf-8") as f:
                content = json.load(f)
                self.surahs = content.get("surahs", {})
                self.ayat = content.get("ayat", {})
        else:
            self.surahs = {}
            self.ayat = {}

    def get_surah_info(self, surah: int) -> Optional[Dict[str, Any]]:
        return self.surahs.get(str(surah))

    def get_ayah_count(self, surah: int) -> Optional[int]:
        info = self.get_surah_info(surah)
        return info["ayahs_count"] if info else None

    def get_ayah(self, surah: int, ayah: int) -> Optional[Dict[str, str]]:
        key = f"{surah}:{ayah}"
        return self.ayat.get(key)

    def has_hadith_corpus(self) -> bool:
        # Stub accessor for compatibility with Issue #24
        return False


# Shared instance across the application
corpus = QuranCorpus()

