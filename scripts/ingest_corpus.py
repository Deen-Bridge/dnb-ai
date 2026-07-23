#!/usr/bin/env python3
"""Download, normalize, and ingest a Quran translation and hadith collection
into the ChromaDB vector store.

Run offline (not at request time):
    python scripts/ingest_corpus.py

Sources
-------
Quran: Saheeh International translation via Tanzil.net.
       Tanzil Quran text © Tanzil.net. License: Creative Commons BY-ND 3.0
       https://tanzil.net/docs/license

Hadith: Sahih al-Bukhari (English) from a permissively licensed GitHub export.
        Source: https://github.com/niwla2305/hadiths
        License: CC0 / Public Domain (per the repository)

Output: ChromaDB persistent index under CHROMA_PERSIST_DIR (default chroma_data/).
        Raw JSONL will be cached under data/.
"""

import argparse
import json
import logging
import os
import sys
import urllib.error
import urllib.request

# Ensure the project root is on sys.path so `rag` can be imported.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag import ChromaStore  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# ---------------------------------------------------------------------------
# Quran: Saheeh International from Tanzil.net
# ---------------------------------------------------------------------------

QURAN_URL = (
    "https://tanzil.net/trans/en.sahih"
)
QURAN_JSONL = os.path.join(DATA_DIR, "quran_en_sahih.jsonl")

SURAH_NAMES = [
    "Al-Fatiha", "Al-Baqarah", "Aal-e-Imran", "An-Nisa", "Al-Ma'idah",
    "Al-An'am", "Al-A'raf", "Al-Anfal", "At-Tawbah", "Yunus",
    "Hud", "Yusuf", "Ar-Ra'd", "Ibrahim", "Al-Hijr",
    "An-Nahl", "Al-Isra", "Al-Kahf", "Maryam", "Ta-Ha",
    "Al-Anbiya", "Al-Hajj", "Al-Mu'minun", "An-Nur", "Al-Furqan",
    "Ash-Shu'ara", "An-Naml", "Al-Qasas", "Al-Ankabut", "Ar-Rum",
    "Luqman", "As-Sajdah", "Al-Ahzab", "Saba", "Fatir",
    "Ya-Sin", "As-Saffat", "Sad", "Az-Zumar", "Ghafir",
    "Fussilat", "Ash-Shura", "Az-Zukhruf", "Ad-Dukhan", "Al-Jathiyah",
    "Al-Ahqaf", "Muhammad", "Al-Fath", "Al-Hujurat", "Qaf",
    "Adh-Dhariyat", "At-Tur", "An-Najm", "Al-Qamar", "Ar-Rahman",
    "Al-Waqi'ah", "Al-Hadid", "Al-Mujadilah", "Al-Hashr", "Al-Mumtahanah",
    "As-Saff", "Al-Jumu'ah", "Al-Munafiqun", "At-Taghabun", "At-Talaq",
    "At-Tahrim", "Al-Mulk", "Al-Qalam", "Al-Haqqah", "Al-Ma'arij",
    "Nuh", "Al-Jinn", "Al-Muzzammil", "Al-Muddaththir", "Al-Qiyamah",
    "Al-Insan", "Al-Mursalat", "An-Naba", "An-Nazi'at", "Abasa",
    "At-Takwir", "Al-Infitar", "Al-Mutaffifin", "Al-Inshiqaq", "Al-Buruj",
    "At-Tariq", "Al-A'la", "Al-Ghashiyah", "Al-Fajr", "Al-Balad",
    "Ash-Shams", "Al-Layl", "Ad-Duhaa", "Ash-Sharh", "At-Tin",
    "Al-Alaq", "Al-Qadr", "Al-Bayyinah", "Az-Zalzalah", "Al-Adiyat",
    "Al-Qari'ah", "At-Takathur", "Al-Asr", "Al-Humazah", "Al-Fil",
    "Quraysh", "Al-Ma'un", "Al-Kawthar", "Al-Kafirun", "An-Nasr",
    "Al-Masad", "Al-Ikhlas", "Al-Falaq", "An-Nas",
]


def download_quran() -> list[dict]:
    """Download the Saheeh International translation from Tanzil.net.

    Tanzil provides a simple format: one line per ayah:
        surah|ayah|text
    """
    logger.info("Downloading Quran translation from %s", QURAN_URL)
    try:
        resp = urllib.request.urlopen(QURAN_URL, timeout=30)
        raw = resp.read().decode("utf-8")
    except Exception as exc:
        logger.error("Failed to download Quran: %s", exc)
        return []

    ayat: list[dict] = []
    for line in raw.strip().split("\n"):
        line = line.strip()
        if not line or "|" not in line:
            continue
        parts = line.split("|", 2)
        if len(parts) < 3:
            continue
        surah_str, ayah_str, text = parts
        try:
            surah = int(surah_str)
            ayah = int(ayah_str)
        except ValueError:
            continue
        surah_name = SURAH_NAMES[surah - 1] if 1 <= surah <= 114 else f"Surah {surah}"
        ayat.append({
            "source": "quran",
            "surah": surah,
            "surah_name": surah_name,
            "ayah": ayah,
            "text": text.strip(),
            "reference": f"Quran {surah}:{ayah}",
        })
    return ayat


# ---------------------------------------------------------------------------
# Hadith: Sahih al-Bukhari
# ---------------------------------------------------------------------------

HADITH_URL = (
    "https://raw.githubusercontent.com/niwla2305/hadiths/main/hadiths/bukhari.json"
)
HADITH_JSONL = os.path.join(DATA_DIR, "hadith_bukhari.jsonl")


def download_hadith() -> list[dict]:
    """Download Sahih al-Bukhari hadith from GitHub."""
    logger.info("Downloading hadith from %s", HADITH_URL)
    try:
        resp = urllib.request.urlopen(HADITH_URL, timeout=30)
        data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        logger.error("Failed to download hadith: %s", exc)
        return []

    entries: list[dict] = []
    for item in data:
        if isinstance(item, dict):
            text = item.get("hadith_english", "") or item.get("text", "")
            if not text:
                continue
            entries.append({
                "source": "hadith",
                "collection": "Sahih al-Bukhari",
                "book": item.get("book", ""),
                "hadith_number": item.get("hadith_number", item.get("refno", "")),
                "text": text.strip(),
                "reference": (
                    f"Sahih al-Bukhari, Book {item.get('book', '?')}, "
                    f"Hadith {item.get('hadith_number', item.get('refno', '?'))}"
                ),
            })
    return entries


# ---------------------------------------------------------------------------
# Normalize & ingest
# ---------------------------------------------------------------------------


def save_jsonl(entries: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logger.info("Saved %d entries to %s", len(entries), path)


def ingest(entries: list[dict], store: ChromaStore) -> None:
    texts = [e["text"] for e in entries]
    metadatas = [
        {k: v for k, v in e.items() if k != "text"} for e in entries
    ]
    ids = [
        f"{e['source']}-{e.get('surah', e.get('hadith_number', idx))}"
        for idx, e in enumerate(entries)
    ]
    store.add_documents(texts, metadatas, ids)
    logger.info("Ingested %d documents into Chroma", len(texts))


def main():
    parser = argparse.ArgumentParser(description="Ingest Islamic corpus into ChromaDB")
    parser.add_argument(
        "--reuse-cache",
        action="store_true",
        help="Reuse cached JSONL files in data/ instead of re-downloading",
    )
    args = parser.parse_args()

    store = ChromaStore()

    # Quran
    if args.reuse_cache and os.path.exists(QURAN_JSONL):
        logger.info("Loading cached Quran from %s", QURAN_JSONL)
        with open(QURAN_JSONL) as f:
            quran_entries = [json.loads(line) for line in f]
    else:
        quran_entries = download_quran()
        if quran_entries:
            save_jsonl(quran_entries, QURAN_JSONL)

    if quran_entries:
        ingest(quran_entries, store)
    else:
        logger.warning("No Quran entries to ingest")

    # Hadith
    if args.reuse_cache and os.path.exists(HADITH_JSONL):
        logger.info("Loading cached hadith from %s", HADITH_JSONL)
        with open(HADITH_JSONL) as f:
            hadith_entries = [json.loads(line) for line in f]
    else:
        hadith_entries = download_hadith()
        if hadith_entries:
            save_jsonl(hadith_entries, HADITH_JSONL)

    if hadith_entries:
        ingest(hadith_entries, store)
    else:
        logger.warning("No hadith entries to ingest")

    logger.info("Ingestion complete. Total documents in store: %d", store.count)


if __name__ == "__main__":
    main()
