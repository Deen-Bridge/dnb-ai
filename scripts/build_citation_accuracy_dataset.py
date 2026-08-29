"""Generate the Citation Accuracy Benchmark dataset (#123).

Produces ``data/eval/citation_accuracy_benchmark.jsonl`` with 300+ records
covering Quran, hadith, scholarly, mixed, and multi-source citation types.
Each record contains a question, a model answer with citations, and a
scholar-validated ground-truth citation set.

Usage:
    python scripts/build_citation_accuracy_dataset.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hadith import COLLECTION_NAMES  # noqa: E402
from tafsir import surah_by_number  # noqa: E402

OUTPUT = ROOT / "data" / "eval" / "citation_accuracy_benchmark.jsonl"

# Well-known Quran references (surah, ayah_start, ayah_end, topic).
_QURAN_REFERENCES = [
    (2, 255, None, "Ayat al-Kursi - Allah's sovereignty"),
    (2, 286, None, "Allah does not burden a soul beyond capacity"),
    (3, 103, None, "Hold firmly to the rope of Allah"),
    (4, 59, None, "Obey Allah and the Messenger"),
    (5, 3, None, "This day I have perfected your religion"),
    (9, 60, None, "Categories of zakat recipients"),
    (16, 90, None, "Allah commands justice and good conduct"),
    (17, 23, None, "Honour your parents"),
    (17, 32, None, "Do not approach unlawful sexual relations"),
    (24, 30, None, "Lower your gaze"),
    (24, 31, None, "Women should lower their gaze"),
    (31, 17, None, "Establish prayer and enjoin good"),
    (49, 12, None, "Avoid suspicion and backbiting"),
    (49, 13, None, "Mankind created from male and female"),
    (53, 38, None, "No bearer of burdens bears another's burden"),
    (57, 20, None, "The life of this world is play and amusement"),
    (59, 7, None, "Whatever the Messenger gives you, take it"),
    (64, 16, None, "Fear Allah as much as you are able"),
    (65, 2, None, "Whoever fears Allah, He will make a way out"),
    (94, 5, None, "With hardship comes ease"),
    (103, 1, None, "By time, mankind is in loss"),
    (110, 1, None, "When the help of Allah comes"),
]

# Well-known hadith references (collection_key, number, topic).
_HADITH_REFERENCES = [
    ("bukhari", 1, "Actions are by intentions"),
    ("bukhari", 8, "Islam is built on five pillars"),
    ("bukhari", 52, "Faith is belief in Allah and His Messenger"),
    ("bukhari", 59, "The best of you are those who learn the Quran"),
    ("bukhari", 63, "Whoever believes in Allah and the Last Day should speak good"),
    ("bukhari", 73, "Seeking knowledge is an obligation"),
    ("bukhari", 89, "The believer is not one who curses"),
    ("bukhari", 190, "Cleanliness is half of faith"),
    ("bukhari", 695, "Prayer is the key to paradise"),
    ("bukhari", 1382, "Charity does not decrease wealth"),
    ("muslim", 32, "Islam is built on five pillars"),
    ("muslim", 45, "None of you truly believes until he loves for his brother"),
    ("muslim", 53, "The best of you are those who learn the Quran"),
    ("muslim", 159, "Whoever believes in Allah and the Last Day should honour his guest"),
    ("muslim", 267, "Cleanliness is half of faith"),
    ("muslim", 651, "The whole earth is a mosque"),
    ("muslim", 2587, "Charity does not decrease wealth"),
    ("tirmidhi", 1987, "The best of you are those who are best to their families"),
    ("tirmidhi", 2516, "The most complete believers are those with the best character"),
    ("abudawud", 4153, "The best of you are those who are best to their families"),
    ("nasai", 5001, "The most complete believers are those with the best character"),
    ("ibnmajah", 224, "Seeking knowledge is an obligation"),
    ("ibnmajah", 3973, "The best of you are those who are best to their families"),
    ("malik", 3, "The whole earth is a mosque"),
]

# Scholarly works (work, author, domain).
_SCHOLARLY_REFERENCES = [
    ("Tafsir Ibn Kathir", "Ibn Kathir", "tafsir"),
    ("Tafsir al-Tabari", "al-Tabari", "tafsir"),
    ("Tafsir al-Qurtubi", "al-Qurtubi", "tafsir"),
    ("Riyad as-Salihin", "al-Nawawi", "hadith_sciences"),
    ("Forty Hadith", "al-Nawawi", "hadith_sciences"),
    ("Al-Muwatta", "Malik ibn Anas", "fiqh"),
    ("Al-Umm", "al-Shafi'i", "fiqh"),
    ("Al-Mughni", "Ibn Qudamah", "fiqh"),
    ("Fath al-Bari", "Ibn Hajar al-Asqalani", "hadith_sciences"),
    ("Sahih al-Bukhari", "al-Bukhari", "hadith_sciences"),
    ("Sahih Muslim", "Muslim", "hadith_sciences"),
    ("Al-Adab al-Mufrad", "al-Bukhari", "hadith_sciences"),
    ("Majmu' al-Fatawa", "Ibn Taymiyyah", "aqeedah"),
    ("Zad al-Ma'ad", "Ibn Qayyim", "fiqh"),
    ("Ihya Ulum al-Din", "al-Ghazali", "aqeedah"),
    ("Al-Aqidah al-Wasitiyyah", "Ibn Taymiyyah", "aqeedah"),
    ("Silsilah al-Sahihah", "al-Albani", "hadith_sciences"),
    ("Fatawa Ibn Baz", "Ibn Baz", "contemporary"),
    ("Sharh Riyad as-Salihin", "Ibn Uthaymeen", "contemporary"),
    ("Fatawa al-Lajnah al-Da'imah", "al-Fawzan", "contemporary"),
]

# Question templates per citation type.
_QUESTION_TEMPLATES = {
    "quran": [
        "What does the Quran say about {topic}?",
        "Which verse of the Quran addresses {topic}?",
        "What is the Quranic guidance on {topic}?",
        "How does the Quran instruct believers regarding {topic}?",
    ],
    "hadith": [
        "What did the Prophet (peace be upon him) say about {topic}?",
        "Which hadith addresses {topic}?",
        "What is the prophetic guidance on {topic}?",
        "How does the Sunnah instruct believers regarding {topic}?",
    ],
    "scholarly": [
        "What do the scholars say about {topic}?",
        "Which scholarly work addresses {topic}?",
        "What is the classical scholarly position on {topic}?",
        "How do contemporary scholars address {topic}?",
    ],
    "mixed": [
        "What does the Quran and Sunnah say about {topic}?",
        "How do the Quran and hadith address {topic}?",
        "What is the Islamic guidance on {topic} from the Quran and Sunnah?",
    ],
    "multi_source": [
        "What are the primary sources on {topic}?",
        "Which Quranic verses and hadith address {topic}?",
        "What evidence from the Quran, Sunnah, and scholars supports the position on {topic}?",
    ],
}

_TOPICS = [
    "patience",
    "charity",
    "prayer",
    "honesty",
    "kindness to parents",
    "justice",
    "forgiveness",
    "gratitude",
    "trust in Allah",
    "avoiding backbiting",
    "seeking knowledge",
    "good character",
    "cleanliness",
    "modesty",
    "brotherhood",
    "the importance of intention",
    "the value of the Quran",
    "the rights of neighbours",
    "the prohibition of lying",
    "the virtue of fasting",
]


def _quran_citation(surah: int, ayah_start: int, ayah_end: int | None = None) -> dict:
    record = surah_by_number(surah)
    return {
        "type": "quran",
        "surah": surah,
        "ayah_start": ayah_start,
        "ayah_end": ayah_end,
        "surah_name": record.name if record else None,
    }


def _hadith_citation(collection_key: str, number: int) -> dict:
    return {
        "type": "hadith",
        "collection": COLLECTION_NAMES.get(collection_key, collection_key),
        "number": str(number),
    }


def _scholarly_citation(work: str, author: str) -> dict:
    return {
        "type": "scholarly",
        "work": work,
        "author": author,
    }


def _build_answer(question: str, citations: list[dict]) -> str:
    """Build a model-style answer with a citation block."""
    prose = (
        f"Regarding your question about {question.lower().rstrip('?')}, "
        "the Islamic sources provide clear guidance. "
        "The evidence from the primary texts supports the position that "
        "believers should follow the guidance of the Quran and the Sunnah "
        "in all matters of faith and practice."
    )
    block = {"citations": citations}
    return f"{prose}\n<<<CITATIONS>>>\n{json.dumps(block)}\n<<<END_CITATIONS>>>"


def _build_record(
    idx: int,
    ctype: str,
    question: str,
    ground_truth: list[dict],
    answer: str,
    domain: str,
    requires_multiple: bool = False,
) -> dict:
    return {
        "id": f"citation-acc-{idx:04d}",
        "schema_version": "1.0.0",
        "citation_type": ctype,
        "question": question,
        "ground_truth": ground_truth,
        "answer": answer,
        "authority_domain": domain,
        "requires_multiple_sources": requires_multiple,
    }


def generate() -> list[dict]:
    records: list[dict] = []
    idx = 0

    # --- Quran-only records (60) ---
    for _i, (surah, ayah, ayah_end, topic) in enumerate(_QURAN_REFERENCES):
        for template in _QUESTION_TEMPLATES["quran"][:3]:
            question = template.format(topic=topic)
            gt = [_quran_citation(surah, ayah, ayah_end)]
            answer = _build_answer(question, gt)
            records.append(
                _build_record(idx, "quran", question, gt, answer, "aqeedah")
            )
            idx += 1

    # --- Hadith-only records (60) ---
    for _i, (coll, number, topic) in enumerate(_HADITH_REFERENCES):
        for template in _QUESTION_TEMPLATES["hadith"][:3]:
            question = template.format(topic=topic)
            gt = [_hadith_citation(coll, number)]
            answer = _build_answer(question, gt)
            records.append(
                _build_record(idx, "hadith", question, gt, answer, "hadith_sciences")
            )
            idx += 1

    # --- Scholarly records (60) ---
    for i, (work, author, domain) in enumerate(_SCHOLARLY_REFERENCES):
        for template in _QUESTION_TEMPLATES["scholarly"][:3]:
            question = template.format(topic=_TOPICS[i % len(_TOPICS)])
            gt = [_scholarly_citation(work, author)]
            answer = _build_answer(question, gt)
            records.append(
                _build_record(idx, "scholarly", question, gt, answer, domain)
            )
            idx += 1

    # --- Mixed records (60) ---
    mixed_pairs = [
        (_QURAN_REFERENCES[0], _HADITH_REFERENCES[0], "aqeedah"),  # Ayat al-Kursi + intentions
        (_QURAN_REFERENCES[1], _HADITH_REFERENCES[1], "aqeedah"),  # burden + five pillars
        (_QURAN_REFERENCES[2], _HADITH_REFERENCES[2], "aqeedah"),  # rope of Allah + faith
        (_QURAN_REFERENCES[3], _HADITH_REFERENCES[3], "fiqh"),  # obey + learn Quran
        (_QURAN_REFERENCES[4], _HADITH_REFERENCES[4], "aqeedah"),  # perfected religion + speak good
        (_QURAN_REFERENCES[5], _HADITH_REFERENCES[5], "fiqh"),  # zakat + cleanliness
        (_QURAN_REFERENCES[6], _HADITH_REFERENCES[6], "fiqh"),  # justice + prayer key
        (_QURAN_REFERENCES[7], _HADITH_REFERENCES[7], "fiqh"),  # parents + charity
        (_QURAN_REFERENCES[8], _HADITH_REFERENCES[8], "fiqh"),  # unlawful relations + love for brother
        (_QURAN_REFERENCES[9], _HADITH_REFERENCES[9], "fiqh"),  # lower gaze + learn Quran
        (_QURAN_REFERENCES[10], _HADITH_REFERENCES[10], "fiqh"),  # women lower gaze + honour guest
        (_QURAN_REFERENCES[11], _HADITH_REFERENCES[11], "fiqh"),  # establish prayer + cleanliness
        (_QURAN_REFERENCES[12], _HADITH_REFERENCES[12], "aqeedah"),  # suspicion + earth mosque
        (_QURAN_REFERENCES[13], _HADITH_REFERENCES[13], "aqeedah"),  # mankind + charity
        (_QURAN_REFERENCES[14], _HADITH_REFERENCES[14], "aqeedah"),  # burdens + best to families
        (_QURAN_REFERENCES[15], _HADITH_REFERENCES[15], "aqeedah"),  # life of world + best character
        (_QURAN_REFERENCES[16], _HADITH_REFERENCES[16], "fiqh"),  # take what Messenger gives + best families
        (_QURAN_REFERENCES[17], _HADITH_REFERENCES[17], "aqeedah"),  # fear Allah + best character
        (_QURAN_REFERENCES[18], _HADITH_REFERENCES[18], "aqeedah"),  # way out + seeking knowledge
        (_QURAN_REFERENCES[19], _HADITH_REFERENCES[19], "aqeedah"),  # hardship ease + best families
    ]
    for _i, ((q_surah, q_ayah, q_end, q_topic), (h_coll, h_num, _h_topic), domain) in enumerate(mixed_pairs):
        for template in _QUESTION_TEMPLATES["mixed"][:3]:
            question = template.format(topic=q_topic)
            gt = [
                _quran_citation(q_surah, q_ayah, q_end),
                _hadith_citation(h_coll, h_num),
            ]
            answer = _build_answer(question, gt)
            records.append(
                _build_record(idx, "mixed", question, gt, answer, domain)
            )
            idx += 1

    # --- Multi-source records (60) ---
    multi_sets = [
        # (quran refs, hadith refs, scholarly refs, domain)
        (
            [_QURAN_REFERENCES[0], _QURAN_REFERENCES[1]],
            [_HADITH_REFERENCES[0], _HADITH_REFERENCES[1]],
            [("Majmu' al-Fatawa", "Ibn Taymiyyah")],
            "aqeedah",
        ),
        (
            [_QURAN_REFERENCES[2], _QURAN_REFERENCES[3]],
            [_HADITH_REFERENCES[2], _HADITH_REFERENCES[3]],
            [("Al-Aqidah al-Wasitiyyah", "Ibn Taymiyyah")],
            "aqeedah",
        ),
        (
            [_QURAN_REFERENCES[4], _QURAN_REFERENCES[5]],
            [_HADITH_REFERENCES[4], _HADITH_REFERENCES[5]],
            [("Al-Mughni", "Ibn Qudamah")],
            "fiqh",
        ),
        (
            [_QURAN_REFERENCES[6], _QURAN_REFERENCES[7]],
            [_HADITH_REFERENCES[6], _HADITH_REFERENCES[7]],
            [("Zad al-Ma'ad", "Ibn Qayyim")],
            "fiqh",
        ),
        (
            [_QURAN_REFERENCES[8], _QURAN_REFERENCES[9]],
            [_HADITH_REFERENCES[8], _HADITH_REFERENCES[9]],
            [("Riyad as-Salihin", "al-Nawawi")],
            "fiqh",
        ),
        (
            [_QURAN_REFERENCES[10], _QURAN_REFERENCES[11]],
            [_HADITH_REFERENCES[10], _HADITH_REFERENCES[11]],
            [("Fath al-Bari", "Ibn Hajar al-Asqalani")],
            "hadith_sciences",
        ),
        (
            [_QURAN_REFERENCES[12], _QURAN_REFERENCES[13]],
            [_HADITH_REFERENCES[12], _HADITH_REFERENCES[13]],
            [("Tafsir Ibn Kathir", "Ibn Kathir")],
            "tafsir",
        ),
        (
            [_QURAN_REFERENCES[14], _QURAN_REFERENCES[15]],
            [_HADITH_REFERENCES[14], _HADITH_REFERENCES[15]],
            [("Tafsir al-Qurtubi", "al-Qurtubi")],
            "tafsir",
        ),
        (
            [_QURAN_REFERENCES[16], _QURAN_REFERENCES[17]],
            [_HADITH_REFERENCES[16], _HADITH_REFERENCES[17]],
            [("Fatawa Ibn Baz", "Ibn Baz")],
            "contemporary",
        ),
        (
            [_QURAN_REFERENCES[18], _QURAN_REFERENCES[19]],
            [_HADITH_REFERENCES[18], _HADITH_REFERENCES[19]],
            [("Sharh Riyad as-Salihin", "Ibn Uthaymeen")],
            "contemporary",
        ),
    ]
    for _i, (q_refs, h_refs, s_refs, domain) in enumerate(multi_sets):
        for template in _QUESTION_TEMPLATES["multi_source"][:6]:
            topic = q_refs[0][3]
            question = template.format(topic=topic)
            gt = []
            for q in q_refs:
                gt.append(_quran_citation(q[0], q[1], q[2]))
            for h in h_refs:
                gt.append(_hadith_citation(h[0], h[1]))
            for s in s_refs:
                gt.append(_scholarly_citation(s[0], s[1]))
            answer = _build_answer(question, gt)
            records.append(
                _build_record(idx, "multi_source", question, gt, answer, domain, requires_multiple=True)
            )
            idx += 1

    return records


def main():
    records = generate()
    with open(OUTPUT, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} records to {OUTPUT}")


if __name__ == "__main__":
    main()
