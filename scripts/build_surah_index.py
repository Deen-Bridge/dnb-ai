"""Build ``data/quran/surah_index.json`` — the surah reference table.

The table carries no Quranic text: only each surah's number, transliterated
name, Arabic name, revelation place, and ayah count under the Kufan
numbering (the counting used by the Uthmani mushaf and by every mainstream
tafsir source). It exists so ayah references can be validated offline —
"surah 115" and "2:300" are rejected before any network call is made, and
before a model ever gets the chance to invent a verse.

Run from the repository root:

    python scripts/build_surah_index.py

The script is deterministic and takes no network access; it is checked in so
the table's provenance is reviewable rather than a wall of magic numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "quran" / "surah_index.json"

# (transliterated name, Arabic name, revelation place, ayah count)
SURAHS: list[tuple[str, str, str, int]] = [
    ("Al-Fatihah", "الفاتحة", "meccan", 7),
    ("Al-Baqarah", "البقرة", "medinan", 286),
    ("Ali 'Imran", "آل عمران", "medinan", 200),
    ("An-Nisa", "النساء", "medinan", 176),
    ("Al-Ma'idah", "المائدة", "medinan", 120),
    ("Al-An'am", "الأنعام", "meccan", 165),
    ("Al-A'raf", "الأعراف", "meccan", 206),
    ("Al-Anfal", "الأنفال", "medinan", 75),
    ("At-Tawbah", "التوبة", "medinan", 129),
    ("Yunus", "يونس", "meccan", 109),
    ("Hud", "هود", "meccan", 123),
    ("Yusuf", "يوسف", "meccan", 111),
    ("Ar-Ra'd", "الرعد", "medinan", 43),
    ("Ibrahim", "إبراهيم", "meccan", 52),
    ("Al-Hijr", "الحجر", "meccan", 99),
    ("An-Nahl", "النحل", "meccan", 128),
    ("Al-Isra", "الإسراء", "meccan", 111),
    ("Al-Kahf", "الكهف", "meccan", 110),
    ("Maryam", "مريم", "meccan", 98),
    ("Taha", "طه", "meccan", 135),
    ("Al-Anbiya", "الأنبياء", "meccan", 112),
    ("Al-Hajj", "الحج", "medinan", 78),
    ("Al-Mu'minun", "المؤمنون", "meccan", 118),
    ("An-Nur", "النور", "medinan", 64),
    ("Al-Furqan", "الفرقان", "meccan", 77),
    ("Ash-Shu'ara", "الشعراء", "meccan", 227),
    ("An-Naml", "النمل", "meccan", 93),
    ("Al-Qasas", "القصص", "meccan", 88),
    ("Al-'Ankabut", "العنكبوت", "meccan", 69),
    ("Ar-Rum", "الروم", "meccan", 60),
    ("Luqman", "لقمان", "meccan", 34),
    ("As-Sajdah", "السجدة", "meccan", 30),
    ("Al-Ahzab", "الأحزاب", "medinan", 73),
    ("Saba", "سبأ", "meccan", 54),
    ("Fatir", "فاطر", "meccan", 45),
    ("Ya-Sin", "يس", "meccan", 83),
    ("As-Saffat", "الصافات", "meccan", 182),
    ("Sad", "ص", "meccan", 88),
    ("Az-Zumar", "الزمر", "meccan", 75),
    ("Ghafir", "غافر", "meccan", 85),
    ("Fussilat", "فصلت", "meccan", 54),
    ("Ash-Shura", "الشورى", "meccan", 53),
    ("Az-Zukhruf", "الزخرف", "meccan", 89),
    ("Ad-Dukhan", "الدخان", "meccan", 59),
    ("Al-Jathiyah", "الجاثية", "meccan", 37),
    ("Al-Ahqaf", "الأحقاف", "meccan", 35),
    ("Muhammad", "محمد", "medinan", 38),
    ("Al-Fath", "الفتح", "medinan", 29),
    ("Al-Hujurat", "الحجرات", "medinan", 18),
    ("Qaf", "ق", "meccan", 45),
    ("Adh-Dhariyat", "الذاريات", "meccan", 60),
    ("At-Tur", "الطور", "meccan", 49),
    ("An-Najm", "النجم", "meccan", 62),
    ("Al-Qamar", "القمر", "meccan", 55),
    ("Ar-Rahman", "الرحمن", "medinan", 78),
    ("Al-Waqi'ah", "الواقعة", "meccan", 96),
    ("Al-Hadid", "الحديد", "medinan", 29),
    ("Al-Mujadila", "المجادلة", "medinan", 22),
    ("Al-Hashr", "الحشر", "medinan", 24),
    ("Al-Mumtahanah", "الممتحنة", "medinan", 13),
    ("As-Saff", "الصف", "medinan", 14),
    ("Al-Jumu'ah", "الجمعة", "medinan", 11),
    ("Al-Munafiqun", "المنافقون", "medinan", 11),
    ("At-Taghabun", "التغابن", "medinan", 18),
    ("At-Talaq", "الطلاق", "medinan", 12),
    ("At-Tahrim", "التحريم", "medinan", 12),
    ("Al-Mulk", "الملك", "meccan", 30),
    ("Al-Qalam", "القلم", "meccan", 52),
    ("Al-Haqqah", "الحاقة", "meccan", 52),
    ("Al-Ma'arij", "المعارج", "meccan", 44),
    ("Nuh", "نوح", "meccan", 28),
    ("Al-Jinn", "الجن", "meccan", 28),
    ("Al-Muzzammil", "المزمل", "meccan", 20),
    ("Al-Muddaththir", "المدثر", "meccan", 56),
    ("Al-Qiyamah", "القيامة", "meccan", 40),
    ("Al-Insan", "الإنسان", "medinan", 31),
    ("Al-Mursalat", "المرسلات", "meccan", 50),
    ("An-Naba", "النبأ", "meccan", 40),
    ("An-Nazi'at", "النازعات", "meccan", 46),
    ("'Abasa", "عبس", "meccan", 42),
    ("At-Takwir", "التكوير", "meccan", 29),
    ("Al-Infitar", "الانفطار", "meccan", 19),
    ("Al-Mutaffifin", "المطففين", "meccan", 36),
    ("Al-Inshiqaq", "الانشقاق", "meccan", 25),
    ("Al-Buruj", "البروج", "meccan", 22),
    ("At-Tariq", "الطارق", "meccan", 17),
    ("Al-A'la", "الأعلى", "meccan", 19),
    ("Al-Ghashiyah", "الغاشية", "meccan", 26),
    ("Al-Fajr", "الفجر", "meccan", 30),
    ("Al-Balad", "البلد", "meccan", 20),
    ("Ash-Shams", "الشمس", "meccan", 15),
    ("Al-Layl", "الليل", "meccan", 21),
    ("Ad-Duha", "الضحى", "meccan", 11),
    ("Ash-Sharh", "الشرح", "meccan", 8),
    ("At-Tin", "التين", "meccan", 8),
    ("Al-'Alaq", "العلق", "meccan", 19),
    ("Al-Qadr", "القدر", "meccan", 5),
    ("Al-Bayyinah", "البينة", "medinan", 8),
    ("Az-Zalzalah", "الزلزلة", "medinan", 8),
    ("Al-'Adiyat", "العاديات", "meccan", 11),
    ("Al-Qari'ah", "القارعة", "meccan", 11),
    ("At-Takathur", "التكاثر", "meccan", 8),
    ("Al-'Asr", "العصر", "meccan", 3),
    ("Al-Humazah", "الهمزة", "meccan", 9),
    ("Al-Fil", "الفيل", "meccan", 5),
    ("Quraysh", "قريش", "meccan", 4),
    ("Al-Ma'un", "الماعون", "meccan", 7),
    ("Al-Kawthar", "الكوثر", "meccan", 3),
    ("Al-Kafirun", "الكافرون", "meccan", 6),
    ("An-Nasr", "النصر", "medinan", 3),
    ("Al-Masad", "المسد", "meccan", 5),
    ("Al-Ikhlas", "الإخلاص", "meccan", 4),
    ("Al-Falaq", "الفلق", "meccan", 5),
    ("An-Nas", "الناس", "meccan", 6),
]

# Alternative spellings users and models actually type. Kept alongside the
# canonical name so "surah al-asr", "surat ul-Asr" and "asr" all resolve.
ALIASES: dict[int, list[str]] = {
    1: ["fatiha", "fatihah", "opening", "umm al-kitab"],
    2: ["baqara", "baqarah", "cow"],
    3: ["al-imran", "aal-imran", "imran", "family of imran"],
    4: ["nisa", "nisaa", "women"],
    5: ["maidah", "maida", "table spread"],
    6: ["anam", "an'am", "cattle"],
    7: ["araf", "a'raf", "heights"],
    9: ["tawba", "taubah", "bara'ah", "baraah", "repentance"],
    12: ["joseph"],
    14: ["abraham"],
    17: ["bani israil", "isra", "night journey"],
    18: ["kahf", "cave"],
    19: ["mary"],
    20: ["ta-ha", "ta ha"],
    24: ["nur", "noor", "light"],
    32: ["sajda", "prostration"],
    36: ["yasin", "yaseen", "ya sin"],
    41: ["ha mim sajdah"],
    55: ["rahman", "most merciful"],
    56: ["waqia", "waqiah", "inevitable"],
    67: ["mulk", "sovereignty", "dominion"],
    71: ["noah"],
    76: ["dahr", "ad-dahr", "man"],
    78: ["naba", "tidings"],
    93: ["duha", "morning hours"],
    94: ["inshirah", "al-inshirah", "sharh", "relief"],
    97: ["qadr", "power", "decree"],
    103: ["asr", "time", "declining day"],
    108: ["kawthar", "abundance"],
    109: ["kafirun", "disbelievers"],
    110: ["nasr", "divine support"],
    112: ["ikhlas", "sincerity", "tawhid"],
    113: ["falaq", "daybreak"],
    114: ["nas", "mankind"],
}


def build() -> list[dict[str, object]]:
    return [
        {
            "number": i,
            "name": name,
            "arabic_name": arabic,
            "revelation_place": place,
            "ayah_count": count,
            "aliases": ALIASES.get(i, []),
        }
        for i, (name, arabic, place, count) in enumerate(SURAHS, start=1)
    ]


def main() -> None:
    surahs = build()
    assert len(surahs) == 114, f"expected 114 surahs, got {len(surahs)}"
    total = sum(int(s["ayah_count"]) for s in surahs)
    assert total == 6236, f"expected 6236 ayat under Kufan numbering, got {total}"

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(surahs, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(surahs)} surahs ({total} ayat) to {OUT_PATH}")


if __name__ == "__main__":
    main()
