import json
from pathlib import Path
from typing import Any

DATA_PATH = Path(__file__).parent / "data" / "quran_uthmani.json"
ALIGNMENT_DATA_PATH = Path(__file__).parent / "data" / "quran_phoneme_alignments.json"


class QuranCorpus:
    def __init__(self, data_file: Path = DATA_PATH, alignment_data_file: Path = ALIGNMENT_DATA_PATH):
        self.data_file = data_file
        self.alignment_data_file = alignment_data_file
        self.surahs: dict[str, dict[str, Any]] = {}
        self.ayat: dict[str, dict[str, str]] = {}
        self.alignments: dict[str, Any] = {}
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

        if self.alignment_data_file.exists():
            with open(self.alignment_data_file, encoding="utf-8") as f:
                self.alignments = json.load(f)
        else:
            self.alignments = {}

    def get_surah_info(self, surah: int) -> dict[str, Any] | None:
        return self.surahs.get(str(surah))

    def get_ayah_count(self, surah: int) -> int | None:
        info = self.get_surah_info(surah)
        return info["ayahs_count"] if info else None

    def get_ayah(self, surah: int, ayah: int) -> dict[str, str] | None:
        key = f"{surah}:{ayah}"
        return self.ayat.get(key)

    def has_hadith_corpus(self) -> bool:
        return False

    def get_phoneme_alignment(self, surah: int, ayah: int) -> dict[str, Any] | None:
        key = f"{surah}:{ayah}"
        return self.alignments.get(key)

    def get_word_timestamps(self, surah: int, ayah: int) -> list[dict[str, Any]] | None:
        alignment = self.get_phoneme_alignment(surah, ayah)
        if alignment:
            return alignment.get("words")
        return None

    def get_alignment_confidence(self, surah: int, ayah: int) -> float | None:
        alignment = self.get_phoneme_alignment(surah, ayah)
        if alignment:
            return alignment.get("confidence")
        return None


# Shared instance across the application
corpus = QuranoCorpus()
