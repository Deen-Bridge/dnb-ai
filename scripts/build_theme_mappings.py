"""Build the theme→verse mapping dataset for the Quranic concordance (#125).

Why a curated table
-------------------
The service deliberately does not bundle the Quran's text (see
``data/quran/PROVENANCE.md``); ayah text is fetched per-request from the
Quran.com API. A keyword-scraped mapping derived from a fetched corpus would
therefore not be reproducible offline, and auto-scraped topical mappings are
exactly the kind of loose association this project refuses to present as
scholarship. Instead the source of truth here is a small, hand-curated table of
well-established theme→verse associations drawn from the classical tafsir and
concordance tradition (e.g. Ayat al-Kursi → tawhid). The script validates every
entry against the bundled surah index (``data/quran/surah_index.json``) so a
typo like ``2:300`` fails the build rather than shipping a fabricated verse.

The script is idempotent: it rewrites ``data/theme_verses.json`` from this
table, so the checked-in dataset and the code can never drift apart. Run it
with ``--check`` in CI to verify the committed file matches the table.

Entry fields
------------
``surah`` / ``ayah``
    The verse, bounds-checked against the 114-surah index.
``theme_id``
    A theme id from ``thematic_quran.py``'s taxonomy (main or sub-theme).
``relevance_score`` (0..1)
    How directly the verse expresses the theme: 1.0 for an archetypal verse,
    0.8–0.95 for clear thematic relevance, 0.5–0.7 for contextual relevance.
``context_type``
    ``primary`` — the verse is a central proof-text for the theme; ``secondary``
    — it bears on the theme in context without being a proof-text.
``annotation``
    A short human-readable note naming why the verse is mapped here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_OUTPUT = ROOT / "data" / "theme_verses.json"
_SURAH_INDEX = ROOT / "data" / "quran" / "surah_index.json"

# ---------------------------------------------------------------------------
# Curated concordance table: (surah, ayah, theme_id, relevance, context, note)
# ---------------------------------------------------------------------------

CURATED_MAPPINGS: list[tuple[int, int, str, float, str, str]] = [
    # ---- tawhid (Monotheism) -------------------------------------------------
    (2, 255, "tawhid", 1.0, "primary", "Ayat al-Kursi — Allah's oneness, life, and sovereignty"),
    (112, 1, "tawhid", 1.0, "primary", "Qul Huwa Allahu Ahad — the essence of tawhid"),
    (112, 2, "tawhid", 1.0, "primary", "Allah us-Samad — the Self-Sufficient"),
    (112, 3, "tawhid", 1.0, "primary", "Neither begets nor is begotten"),
    (112, 4, "tawhid", 1.0, "primary", "None is comparable to Him"),
    (3, 18, "tawhid", 0.95, "primary", "Allah bears witness that there is no deity but Him"),
    (20, 14, "tawhid", 0.95, "primary", "Indeed, I am Allah; there is no deity except Me"),
    (47, 19, "tawhid", 0.9, "primary", "Know that there is no deity except Allah"),
    (37, 35, "tawhid", 0.85, "primary", "There is no deity but Allah"),
    (59, 22, "tawhid", 0.9, "primary", "He is Allah, other than whom there is no deity"),
    (59, 23, "tawhid", 0.85, "primary", "Allah's names and attributes"),
    (6, 102, "tawhid", 0.85, "primary", "That is Allah, your Lord; there is no deity except Him"),
    (2, 163, "tawhid", 0.9, "primary", "Your god is one God"),
    (16, 51, "tawhid", 0.85, "primary", "Do not take two deities; indeed, He is but one God"),
    (21, 25, "tawhid", 0.85, "primary", "There is no deity except Me, so worship Me"),
    (23, 91, "tawhid", 0.8, "secondary", "Allah has not taken any son, nor has there been any deity with Him"),
    (17, 111, "tawhid", 0.8, "secondary", "Praise to Allah who has not taken a son"),
    # ---- tawhid-rububiyyah (Lordship) ----------------------------------------
    (1, 2, "tawhid-rububiyyah", 1.0, "primary", "All praise is due to Allah, Lord of the worlds"),
    (13, 16, "tawhid-rububiyyah", 0.9, "primary", "Say: Allah is the Creator of all things, the One"),
    (39, 62, "tawhid-rububiyyah", 0.9, "primary", "Allah is the Creator of all things"),
    (6, 102, "tawhid-rububiyyah", 0.9, "primary", "No deity except Him, Creator of all things"),
    (35, 3, "tawhid-rububiyyah", 0.85, "primary", "Is there any creator other than Allah?"),
    (10, 31, "tawhid-rububiyyah", 0.85, "primary", "Who provides for you from the heaven and the earth?"),
    (2, 21, "tawhid-rububiyyah", 0.85, "primary", "O mankind, worship your Lord, who created you"),
    (23, 85, "tawhid-rububiyyah", 0.8, "secondary", "To whom belongs the earth and whoever is on it?"),
    # ---- tawhid-uluhiyyah (Worship of Allah alone) ----------------------------
    (20, 14, "tawhid-uluhiyyah", 0.95, "primary", "Worship Me and establish prayer for My remembrance"),
    (51, 56, "tawhid-uluhiyyah", 1.0, "primary", "I did not create jinn and mankind except to worship Me"),
    (2, 21, "tawhid-uluhiyyah", 0.9, "primary", "O mankind, worship your Lord"),
    (98, 5, "tawhid-uluhiyyah", 0.95, "primary", "They were not commanded except to worship Allah alone"),
    (6, 162, "tawhid-uluhiyyah", 0.9, "primary", "My prayer, my rites, my life, and my death are for Allah"),
    (41, 37, "tawhid-uluhiyyah", 0.85, "primary", "Do not prostrate to the sun or to the moon; prostrate to Allah"),
    (16, 36, "tawhid-uluhiyyah", 0.85, "primary", "Worship Allah and avoid Taghut"),
    (21, 25, "tawhid-uluhiyyah", 0.9, "primary", "There is no deity except Me, so worship Me"),
    # ---- tawhid-asma-sifat (Names & Attributes) -------------------------------
    (59, 22, "tawhid-asma-sifat", 0.95, "primary", "He is Allah — the Knower of the unseen and the witnessed"),
    (59, 23, "tawhid-asma-sifat", 0.95, "primary", "The Sovereign, the Pure, the Perfector, the Almighty"),
    (
        59,
        24,
        "tawhid-asma-sifat",
        0.95,
        "primary",
        "He is Allah, the Creator, the Evolver; to Him belong the best names",
    ),
    (20, 8, "tawhid-asma-sifat", 0.9, "primary", "Allah — there is no deity except Him; to Him belong the best names"),
    (7, 180, "tawhid-asma-sifat", 0.9, "primary", "To Allah belong the best names; invoke Him by them"),
    (2, 255, "tawhid-asma-sifat", 0.9, "primary", "The Ever-Living, the Sustainer of all existence"),
    (17, 110, "tawhid-asma-sifat", 0.8, "secondary", "Call upon Allah or call upon the Most Merciful"),
    (57, 3, "tawhid-asma-sifat", 0.85, "primary", "He is the First and the Last, the Most High and the Most Near"),
    (112, 2, "tawhid-asma-sifat", 0.85, "secondary", "Allah us-Samad"),
    # ---- prophethood (Nubuwwah) ----------------------------------------------
    (
        33,
        40,
        "prophethood",
        1.0,
        "primary",
        "Muhammad is not the father of any of your men, but the Messenger of Allah",
    ),
    (3, 144, "prophethood", 0.9, "primary", "Muhammad is not but a messenger"),
    (48, 29, "prophethood", 0.9, "primary", "Muhammad is the Messenger of Allah"),
    (21, 107, "prophethood", 0.95, "primary", "We have not sent you except as a mercy to the worlds"),
    (33, 21, "prophethood", 0.9, "primary", "In the Messenger of Allah you have an excellent example"),
    (3, 164, "prophethood", 0.9, "primary", "Allah conferred a great favor upon the believers by sending a Messenger"),
    (2, 151, "prophethood", 0.9, "primary", "Just as We have sent among you a Messenger from yourselves"),
    (16, 36, "prophethood", 0.85, "primary", "We have sent to every nation a messenger"),
    (10, 47, "prophethood", 0.85, "primary", "For every nation is a messenger"),
    (35, 24, "prophethood", 0.8, "secondary", "There was no nation but that a warner had passed within it"),
    (62, 2, "prophethood", 0.85, "primary", "He has sent among the unlettered a Messenger from themselves"),
    (7, 158, "prophethood", 0.85, "primary", "O mankind, I am the Messenger of Allah to you all"),
    # ---- afterlife (Akhirah) -------------------------------------------------
    (2, 281, "afterlife", 0.95, "primary", "Fear a Day when you will be returned to Allah"),
    (
        3,
        185,
        "afterlife",
        0.95,
        "primary",
        "Every soul will taste death, and you will only be given your compensation on the Day of Resurrection",
    ),
    (21, 47, "afterlife", 0.9, "primary", "We place the scales of justice for the Day of Resurrection"),
    (99, 1, "afterlife", 0.9, "primary", "When the earth is shaken with its [final] earthquake"),
    (99, 6, "afterlife", 0.9, "primary", "So whoever does an atom's weight of good will see it"),
    (101, 1, "afterlife", 0.85, "primary", "The Striking Calamity"),
    (81, 1, "afterlife", 0.85, "primary", "When the sun is wrapped up [in darkness]"),
    (82, 1, "afterlife", 0.85, "primary", "When the sky breaks apart"),
    (
        39,
        68,
        "afterlife",
        0.85,
        "primary",
        "The Horn will be blown, and whoever is in the heavens and earth will fall unconscious",
    ),
    (67, 2, "afterlife", 0.85, "primary", "He who created death and life to test you — which of you is best in deed"),
    (
        18,
        49,
        "afterlife",
        0.85,
        "primary",
        "The record [of deeds] will be placed, and you will see the criminals fearful",
    ),
    (50, 20, "afterlife", 0.85, "primary", "The Horn is blown; that is the Day of the Threat"),
    (36, 51, "afterlife", 0.85, "primary", "The Horn will be blown, and suddenly they will rush from their graves"),
    # ---- afterlife-paradise (Jannah) -----------------------------------------
    (
        2,
        25,
        "afterlife-paradise",
        0.95,
        "primary",
        "Give good tidings to those who believe and do righteous deeds — gardens beneath which rivers flow",
    ),
    (
        3,
        133,
        "afterlife-paradise",
        0.9,
        "primary",
        "Race to forgiveness from your Lord and a Garden as wide as the heavens and the earth",
    ),
    (
        4,
        57,
        "afterlife-paradise",
        0.9,
        "primary",
        "We will admit them to gardens beneath which rivers flow, abiding therein forever",
    ),
    (
        47,
        15,
        "afterlife-paradise",
        0.9,
        "primary",
        "The example of Paradise promised to the righteous — rivers of water, milk, wine, and honey",
    ),
    (76, 12, "afterlife-paradise", 0.9, "primary", "Gardens and silk for the patient"),
    (88, 8, "afterlife-paradise", 0.85, "primary", "Faces that Day will be pleasant, with their effort satisfied"),
    (55, 46, "afterlife-paradise", 0.9, "primary", "For whoever fears the standing of his Lord are two gardens"),
    (
        13,
        23,
        "afterlife-paradise",
        0.85,
        "primary",
        "Gardens of perpetual residence; they will enter them with the righteous among their fathers",
    ),
    (56, 10, "afterlife-paradise", 0.85, "primary", "The forerunners — those are the ones brought near [to Allah]"),
    (
        3,
        15,
        "afterlife-paradise",
        0.85,
        "primary",
        "For the righteous are gardens with their Lord beneath which rivers flow",
    ),
    # ---- afterlife-hellfire (Jahannam) ---------------------------------------
    (2, 24, "afterlife-hellfire", 0.9, "primary", "Fear the Fire whose fuel is people and stones"),
    (3, 131, "afterlife-hellfire", 0.9, "primary", "Fear the Fire prepared for the disbelievers"),
    (
        4,
        56,
        "afterlife-hellfire",
        0.9,
        "primary",
        "Indeed, those who disbelieve in Our verses, We will drive them into a Fire",
    ),
    (9, 63, "afterlife-hellfire", 0.85, "primary", "The Fire of Hell; abide therein forever"),
    (
        22,
        19,
        "afterlife-hellfire",
        0.85,
        "primary",
        "Garments of fire will be cut out for them, and scalding water poured over their heads",
    ),
    (74, 26, "afterlife-hellfire", 0.9, "primary", "I will drive him into Saqar"),
    (78, 21, "afterlife-hellfire", 0.85, "primary", "Indeed, Hell has been lying in wait"),
    (
        66,
        6,
        "afterlife-hellfire",
        0.85,
        "primary",
        "Protect yourselves and your families from a Fire whose fuel is people and stones",
    ),
    (25, 11, "afterlife-hellfire", 0.8, "secondary", "We have prepared for those who deny the Hour a Blaze"),
    # ---- afterlife-judgment (Day of Judgment) --------------------------------
    (1, 4, "afterlife-judgment", 1.0, "primary", "Sovereign of the Day of Recompense"),
    (2, 281, "afterlife-judgment", 0.95, "primary", "Fear a Day when you will be returned to Allah"),
    (36, 51, "afterlife-judgment", 0.85, "secondary", "The Horn will be blown, and they will rush from their graves"),
    (82, 15, "afterlife-judgment", 0.9, "primary", "They will [enter to] burn therein on the Day of Recompense"),
    (
        99,
        6,
        "afterlife-judgment",
        0.9,
        "primary",
        "The people will depart separated into categories to be shown their deeds",
    ),
    (21, 47, "afterlife-judgment", 0.9, "primary", "The scales of justice for the Day of Resurrection"),
    (40, 17, "afterlife-judgment", 0.85, "primary", "Every soul will be recompensed that Day for what it earned"),
    (75, 6, "afterlife-judgment", 0.8, "secondary", "He asks: when is the Day of Resurrection?"),
    # ---- worship (Ibadah) ----------------------------------------------------
    (51, 56, "worship", 0.95, "primary", "I did not create jinn and mankind except to worship Me"),
    (20, 14, "worship", 0.9, "primary", "Establish prayer for My remembrance"),
    (98, 5, "worship", 0.9, "primary", "They were not commanded except to worship Allah, being sincere to Him"),
    (2, 43, "worship", 0.85, "primary", "Establish prayer and give zakah and bow with those who bow"),
    (6, 162, "worship", 0.85, "primary", "My prayer, my rites, my life, and my death are for Allah"),
    (2, 83, "worship", 0.8, "secondary", "Worship none but Allah; be dutiful to parents"),
    (2, 21, "worship", 0.85, "primary", "O mankind, worship your Lord"),
    # ---- worship-salah (Prayer) ----------------------------------------------
    (2, 43, "worship-salah", 1.0, "primary", "Establish prayer and give zakah and bow with those who bow"),
    (2, 238, "worship-salah", 0.95, "primary", "Maintain the prayers and the middle prayer"),
    (4, 103, "worship-salah", 0.95, "primary", "Indeed, prayer has been decreed upon the believers at specified times"),
    (
        17,
        78,
        "worship-salah",
        0.9,
        "primary",
        "Establish prayer at the decline of the sun until the darkness of the night",
    ),
    (20, 14, "worship-salah", 0.9, "primary", "Establish prayer for My remembrance"),
    (
        29,
        45,
        "worship-salah",
        0.95,
        "primary",
        "Recite what has been revealed to you of the Book and establish prayer; prayer prohibits immorality and wrongdoing",
    ),
    (
        87,
        14,
        "worship-salah",
        0.85,
        "primary",
        "He has succeeded who purifies himself and mentions the name of his Lord and prays",
    ),
    (5, 6, "worship-salah", 0.85, "primary", "When you rise to prayer, wash your faces and hands"),
    (2, 45, "worship-salah", 0.85, "primary", "Seek help through patience and prayer"),
    (50, 40, "worship-salah", 0.8, "secondary", "And [part] of the night, exalt Him and after the prostrations"),
    # ---- worship-zakah (Charity) ---------------------------------------------
    (2, 43, "worship-zakah", 0.9, "primary", "Establish prayer and give zakah"),
    (
        2,
        110,
        "worship-zakah",
        0.95,
        "primary",
        "Establish prayer and give zakah; whatever good you put forward for yourselves",
    ),
    (2, 177, "worship-zakah", 0.9, "primary", "Righteousness is to believe in Allah... and give zakah"),
    (
        2,
        277,
        "worship-zakah",
        0.9,
        "primary",
        "Those who believe and do righteous deeds and establish prayer and give zakah",
    ),
    (9, 60, "worship-zakah", 0.95, "primary", "Zakah expenditures are only for the poor and the needy"),
    (9, 103, "worship-zakah", 0.9, "primary", "Take from their wealth a charity by which you purify them"),
    (
        30,
        39,
        "worship-zakah",
        0.85,
        "primary",
        "Whatever you give for interest to increase within the wealth of people will not increase with Allah",
    ),
    (58, 13, "worship-zakah", 0.8, "secondary", "Establish prayer and give zakah and obey Allah and His Messenger"),
    (73, 20, "worship-zakah", 0.8, "secondary", "Establish prayer and give zakah and loan Allah a goodly loan"),
    # ---- worship-sawm (Fasting) ----------------------------------------------
    (
        2,
        183,
        "worship-sawm",
        1.0,
        "primary",
        "O you who believe, fasting has been decreed upon you as it was decreed upon those before you",
    ),
    (
        2,
        184,
        "worship-sawm",
        0.95,
        "primary",
        "A limited number of days; whoever is ill or on a journey, then an equal number of other days",
    ),
    (
        2,
        185,
        "worship-sawm",
        0.95,
        "primary",
        "The month of Ramadan, in which the Quran was revealed; so whoever sights the month, let him fast it",
    ),
    (2, 187, "worship-sawm", 0.9, "primary", "It has been made permissible for you the night preceding fasting"),
    (
        33,
        35,
        "worship-sawm",
        0.85,
        "primary",
        "The men who fast and the women who fast — Allah has prepared forgiveness and a great reward",
    ),
    (
        19,
        26,
        "worship-sawm",
        0.7,
        "secondary",
        "I have vowed a fast to the Most Merciful, so I will not speak today to any human",
    ),
    (66, 5, "worship-sawm", 0.6, "secondary", "Contextual mention of fasting among righteous women"),
    # ---- worship-hajj (Pilgrimage) -------------------------------------------
    (2, 158, "worship-hajj", 0.9, "primary", "Indeed, as-Safa and al-Marwah are among the symbols of Allah"),
    (2, 196, "worship-hajj", 0.95, "primary", "Complete the hajj and the umrah for Allah"),
    (2, 197, "worship-hajj", 0.95, "primary", "Hajj is [during] well-known months"),
    (3, 97, "worship-hajj", 1.0, "primary", "Pilgrimage to the House is a duty upon mankind for Allah"),
    (
        22,
        27,
        "worship-hajj",
        0.9,
        "primary",
        "Proclaim to the people the hajj; they will come to you on foot and on every lean camel",
    ),
    (
        22,
        28,
        "worship-hajj",
        0.85,
        "primary",
        "That they may witness benefits for themselves and mention the name of Allah on known days",
    ),
    (5, 2, "worship-hajj", 0.85, "primary", "Do not violate the rites of Allah or the sacred month"),
    (2, 189, "worship-hajj", 0.8, "secondary", "Enter houses from their doors and fear Allah"),
    # ---- ethics (Akhlaq) -----------------------------------------------------
    (16, 90, "ethics", 0.95, "primary", "Allah commands justice, good conduct, and giving to relatives"),
    (4, 135, "ethics", 0.9, "primary", "Be persistently standing firm in justice, witnesses for Allah"),
    (
        49,
        13,
        "ethics",
        0.9,
        "primary",
        "O mankind, We created you from male and female and made you peoples and tribes that you may know one another",
    ),
    (49, 11, "ethics", 0.85, "primary", "Do not let a people ridicule another people"),
    (49, 12, "ethics", 0.85, "primary", "Avoid much assumption; do not spy or backbite"),
    (17, 23, "ethics", 0.9, "primary", "Your Lord has decreed that you worship none but Him and be dutiful to parents"),
    (2, 83, "ethics", 0.85, "primary", "Speak to people good words"),
    (103, 1, "ethics", 0.9, "primary", "By time — mankind is in loss, except those who believe and do righteous deeds"),
    (33, 70, "ethics", 0.85, "primary", "Speak words of appropriate justice"),
    # ---- ethics-justice (Adl) ------------------------------------------------
    (
        4,
        58,
        "ethics-justice",
        0.95,
        "primary",
        "Allah commands you to render trusts to their owners and judge with justice",
    ),
    (
        4,
        135,
        "ethics-justice",
        0.95,
        "primary",
        "Be persistently standing firm in justice, even against yourselves or parents",
    ),
    (
        5,
        8,
        "ethics-justice",
        0.95,
        "primary",
        "Let not hatred of a people prevent you from being just; be just; it is nearer to righteousness",
    ),
    (16, 90, "ethics-justice", 0.9, "primary", "Allah commands justice and good conduct"),
    (
        49,
        9,
        "ethics-justice",
        0.85,
        "primary",
        "If two factions among the believers fight, make peace between them with justice",
    ),
    (55, 9, "ethics-justice", 0.85, "primary", "Establish weight in justice and do not make deficient the balance"),
    (
        57,
        25,
        "ethics-justice",
        0.85,
        "primary",
        "We sent Our messengers with clear proofs and the Book and the balance that people may maintain justice",
    ),
    (6, 152, "ethics-justice", 0.85, "primary", "When you speak, be just, even if it should be to a near relative"),
    # ---- ethics-patience (Sabr) ----------------------------------------------
    (2, 45, "ethics-patience", 0.95, "primary", "Seek help through patience and prayer"),
    (
        2,
        153,
        "ethics-patience",
        0.95,
        "primary",
        "O you who believe, seek help through patience and prayer; indeed, Allah is with the patient",
    ),
    (
        2,
        155,
        "ethics-patience",
        0.9,
        "primary",
        "We will surely test you with something of fear, hunger, and loss of wealth; give good tidings to the patient",
    ),
    (
        3,
        200,
        "ethics-patience",
        0.9,
        "primary",
        "O you who believe, persevere and endure and remain stationed and fear Allah",
    ),
    (
        16,
        126,
        "ethics-patience",
        0.85,
        "primary",
        "If you punish, punish with an equivalent; but if you are patient, it is better for the patient",
    ),
    (39, 10, "ethics-patience", 0.9, "primary", "The patient will be given their reward without account"),
    (
        90,
        17,
        "ethics-patience",
        0.85,
        "primary",
        "Then he is among those who believed and advised one another to patience",
    ),
    (
        103,
        3,
        "ethics-patience",
        0.9,
        "primary",
        "Except those who believe, do righteous deeds, and advise one another to truth and patience",
    ),
    (
        11,
        11,
        "ethics-patience",
        0.85,
        "primary",
        "Except those who are patient and do righteous deeds; those will have forgiveness and great reward",
    ),
    # ---- ethics-gratitude (Shukr) --------------------------------------------
    (
        2,
        152,
        "ethics-gratitude",
        0.95,
        "primary",
        "Remember Me; I will remember you. Be grateful to Me and do not deny Me",
    ),
    (
        2,
        172,
        "ethics-gratitude",
        0.9,
        "primary",
        "Eat of the good things We have provided for you and be grateful to Allah",
    ),
    (14, 7, "ethics-gratitude", 0.95, "primary", "If you are grateful, I will surely increase you [in favor]"),
    (
        16,
        18,
        "ethics-gratitude",
        0.9,
        "primary",
        "If you should count the favors of Allah, you could not enumerate them",
    ),
    (31, 12, "ethics-gratitude", 0.9, "primary", "Whoever is grateful is grateful for the benefit of himself"),
    (55, 13, "ethics-gratitude", 0.85, "primary", "So which of the favors of your Lord would you deny?"),
    (29, 17, "ethics-gratitude", 0.8, "secondary", "Seek provision from Allah and worship Him and be grateful to Him"),
    # ---- social (Muamalat) ---------------------------------------------------
    (4, 1, "social", 0.95, "primary", "O mankind, fear your Lord who created you from one soul"),
    (
        49,
        13,
        "social",
        0.95,
        "primary",
        "We made you peoples and tribes that you may know one another; the most noble of you is the most righteous",
    ),
    (
        33,
        56,
        "social",
        0.85,
        "primary",
        "Indeed, Allah and His angels send blessings upon the Prophet; send blessings upon him and greet him with peace",
    ),
    (4, 32, "social", 0.8, "secondary", "Do not wish for what Allah has given some of you over others"),
    (
        5,
        2,
        "social",
        0.85,
        "primary",
        "Cooperate in righteousness and piety, and do not cooperate in sin and aggression",
    ),
    (16, 90, "social", 0.85, "primary", "Allah commands justice, good conduct, and giving to relatives"),
    (
        2,
        83,
        "social",
        0.85,
        "primary",
        "Be dutiful to parents, relatives, orphans, and the needy, and speak to people good words",
    ),
    (24, 27, "social", 0.8, "secondary", "Do not enter houses other than your own until you ask permission"),
    (49, 10, "social", 0.85, "primary", "The believers are but brothers, so make peace between your brothers"),
    (
        60,
        8,
        "social",
        0.85,
        "primary",
        "Allah does not forbid you from being righteous and just toward those who have not fought you",
    ),
    (17, 26, "social", 0.85, "primary", "Give the relative his right, and the poor and the traveler"),
    # ---- history (Qisas) -----------------------------------------------------
    (11, 25, "history", 0.9, "primary", "We sent Noah to his people — the story of Nuh"),
    (7, 65, "history", 0.9, "primary", "We sent to 'Ad their brother Hud — the story of Hud"),
    (7, 73, "history", 0.9, "primary", "We sent to Thamud their brother Salih — the story of Salih"),
    (11, 84, "history", 0.9, "primary", "We sent to Madyan their brother Shu'ayb — the story of Shu'ayb"),
    (7, 103, "history", 0.9, "primary", "Then We sent Moses with Our signs to Pharaoh — the story of Musa"),
    (
        15,
        80,
        "history",
        0.85,
        "primary",
        "The companions of al-Hijr denied the messengers — the story of Thamud's stone dwellings",
    ),
    (
        25,
        38,
        "history",
        0.85,
        "primary",
        "And 'Ad and Thamud and the companions of the well and many generations between them",
    ),
    (
        29,
        14,
        "history",
        0.85,
        "primary",
        "We sent Noah to his people, and he remained among them a thousand years minus fifty",
    ),
    (18, 9, "history", 0.85, "primary", "The Companions of the Cave — story of the youths of the cave"),
    (27, 15, "history", 0.85, "primary", "We gave David and Solomon knowledge — the story of Dawud and Sulayman"),
    (12, 4, "history", 0.9, "primary", "When Joseph said to his father: I saw eleven stars — the story of Yusuf"),
    (
        37,
        99,
        "history",
        0.85,
        "primary",
        "And he said: Indeed, I will go to my Lord — the story of Ibrahim's sacrifice",
    ),
    (2, 124, "history", 0.85, "primary", "When his Lord tested Abraham with words — the trials of Ibrahim"),
    (20, 9, "history", 0.85, "primary", "Has the story of Moses reached you?"),
    (79, 15, "history", 0.85, "primary", "Has there reached you the story of Moses?"),
    (85, 17, "history", 0.85, "primary", "Has there reached you the story of the soldiers?"),
    (89, 6, "history", 0.85, "primary", "Did you not consider how your Lord dealt with 'Ad?"),
    (
        105,
        1,
        "history",
        0.85,
        "primary",
        "Have you not considered how your Lord dealt with the companions of the elephant?",
    ),
    # ---- creation (Khalq) ----------------------------------------------------
    (
        2,
        164,
        "creation",
        0.95,
        "primary",
        "Indeed, in the creation of the heavens and the earth and the alternation of night and day are signs",
    ),
    (
        3,
        190,
        "creation",
        0.95,
        "primary",
        "In the creation of the heavens and the earth and the alternation of night and day are signs for those of understanding",
    ),
    (45, 3, "creation", 0.9, "primary", "Indeed, within the heavens and the earth are signs for the believers"),
    (51, 20, "creation", 0.9, "primary", "On the earth are signs for the certain, and in yourselves — do you not see?"),
    (67, 3, "creation", 0.9, "primary", "He who created the seven heavens in layers"),
    (88, 17, "creation", 0.9, "primary", "Do they not look at the camels — how they were created?"),
    (29, 20, "creation", 0.9, "primary", "Travel through the land and observe how He began creation"),
    (
        35,
        27,
        "creation",
        0.85,
        "primary",
        "Do you not see that Allah sends down rain from the sky, producing fruits of varying colors?",
    ),
    (7, 54, "creation", 0.85, "primary", "Your Lord is Allah, who created the heavens and the earth in six days"),
    (41, 11, "creation", 0.85, "primary", "Then He directed Himself to the heaven while it was smoke"),
    (
        30,
        22,
        "creation",
        0.85,
        "primary",
        "Among His signs is the creation of the heavens and the earth and the diversity of your languages and colors",
    ),
    (36, 33, "creation", 0.85, "primary", "A sign for them is the dead earth — We brought it to life"),
    (16, 68, "creation", 0.8, "secondary", "Your Lord inspired the bee to take homes in the mountains"),
    # ---- guidance (Hidayah) --------------------------------------------------
    (1, 6, "guidance", 1.0, "primary", "Guide us to the straight path"),
    (2, 2, "guidance", 0.95, "primary", "This is the Book about which there is no doubt, a guidance for the righteous"),
    (
        2,
        185,
        "guidance",
        0.9,
        "primary",
        "The month of Ramadan in which was revealed the Quran, a guidance for the people",
    ),
    (17, 9, "guidance", 0.9, "primary", "Indeed, this Quran guides to that which is most suitable"),
    (6, 125, "guidance", 0.9, "primary", "Whoever Allah wants to guide — He expands his breast to Islam"),
    (20, 123, "guidance", 0.85, "primary", "Whoever follows My guidance will not go astray nor suffer"),
    (39, 23, "guidance", 0.85, "primary", "Allah has sent down the best statement, a Book consistent in its verses"),
    (
        31,
        2,
        "guidance",
        0.85,
        "primary",
        "These are verses of the wise Book, a guidance and mercy for the doers of good",
    ),
    (
        2,
        38,
        "guidance",
        0.85,
        "primary",
        "When guidance comes to you from Me, whoever follows My guidance will have no fear",
    ),
    (
        10,
        57,
        "guidance",
        0.85,
        "primary",
        "O mankind, there has come to you instruction from your Lord and healing for what is in the breasts",
    ),
    (
        16,
        89,
        "guidance",
        0.85,
        "primary",
        "We have sent down to you the Book as clarification for all things and as guidance and mercy",
    ),
    # ---- law (Shariah) -------------------------------------------------------
    (
        4,
        59,
        "law",
        0.9,
        "primary",
        "O you who believe, obey Allah and obey the Messenger and those in authority among you",
    ),
    (5, 44, "law", 0.9, "primary", "Whoever does not judge by what Allah has revealed — those are the disbelievers"),
    (
        6,
        151,
        "law",
        0.95,
        "primary",
        "Come, I will recite what your Lord has prohibited to you — do not associate anything with Him",
    ),
    (17, 23, "law", 0.9, "primary", "Your Lord has decreed that you worship none but Him and be dutiful to parents"),
    (24, 2, "law", 0.9, "primary", "The fornicatress and fornicator — lash each one of them a hundred lashes"),
    (4, 11, "law", 0.9, "primary", "Allah instructs you concerning your children's inheritance"),
    (
        2,
        229,
        "law",
        0.9,
        "primary",
        "Divorce is twice; then keep them in reasonable honor or release them with kindness",
    ),
    (2, 275, "law", 0.95, "primary", "Allah has permitted trade and has forbidden interest (riba)"),
    (
        5,
        90,
        "law",
        0.9,
        "primary",
        "Intoxicants, gambling, idols, and divining arrows are abomination of Satan's work; avoid them",
    ),
    (4, 29, "law", 0.85, "primary", "Do not consume one another's wealth unjustly"),
    (2, 282, "law", 0.85, "primary", "O you who believe, when you contract a debt for a specified term, write it down"),
    (
        5,
        38,
        "law",
        0.85,
        "primary",
        "The thief, male and female, amputate their hands in recompense for what they committed",
    ),
    (49, 9, "law", 0.8, "secondary", "If two factions fight, make peace between them with justice"),
]

# Annotation for entries whose note field is empty (should not happen).
_FALLBACK_NOTE = "Mapped to the theme by the classical concordance tradition."


def _load_surah_bounds() -> dict[int, int]:
    """Load {surah: ayah_count} from the bundled index; asserts integrity."""
    with open(_SURAH_INDEX, encoding="utf-8") as handle:
        index = json.load(handle)
    surahs = index.get("surahs", index) if isinstance(index, dict) else index
    bounds: dict[int, int] = {}
    for entry in surahs:
        number = entry.get("number") or entry.get("surah")
        count = entry.get("ayahs_count") or entry.get("ayah_count") or entry.get("ayahs")
        if number is not None and count is not None:
            bounds[int(number)] = int(count)
    if len(bounds) != 114:
        raise RuntimeError(f"Surah index has {len(bounds)} surahs; expected 114.")
    if sum(bounds.values()) != 6236:
        raise RuntimeError(f"Surah index totals {sum(bounds.values())} ayat; expected 6236.")
    return bounds


def build_mappings() -> list[dict]:
    """Validate the curated table against the index and emit mapping dicts."""
    bounds = _load_surah_bounds()
    mappings: list[dict] = []
    seen: set[tuple[int, int, str]] = set()
    for surah, ayah, theme_id, relevance, context_type, note in CURATED_MAPPINGS:
        if surah not in bounds:
            raise ValueError(f"Surah {surah} is out of the 1..114 range.")
        if not 1 <= ayah <= bounds[surah]:
            raise ValueError(f"Ayah {surah}:{ayah} exceeds {surah}'s real bound of {bounds[surah]}.")
        if not 0.0 <= relevance <= 1.0:
            raise ValueError(f"Relevance score {relevance} for {surah}:{ayah} is outside 0..1.")
        if context_type not in ("primary", "secondary"):
            raise ValueError(f"Context type {context_type!r} for {surah}:{ayah} is invalid.")
        key = (surah, ayah, theme_id)
        if key in seen:
            raise ValueError(f"Duplicate mapping {surah}:{ayah} → {theme_id}.")
        seen.add(key)
        mappings.append(
            {
                "surah": surah,
                "ayah": ayah,
                "theme_id": theme_id,
                "relevance_score": relevance,
                "annotation": note or _FALLBACK_NOTE,
                "scholarly_notes": None,
                "context_type": context_type,
            }
        )
    return mappings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=_OUTPUT, help="Output JSON path")
    parser.add_argument(
        "--check", action="store_true", help="Verify the committed file matches the table; write nothing"
    )
    args = parser.parse_args()

    mappings = build_mappings()
    payload = {"mappings": mappings}

    if args.check:
        if not args.output.exists():
            raise SystemExit(f"Missing dataset: {args.output}")
        with open(args.output, encoding="utf-8") as handle:
            committed = json.load(handle)
        if committed != payload:
            raise SystemExit("data/theme_verses.json is out of date; run scripts/build_theme_mappings.py")
        print(f"OK: {len(mappings)} mappings match the curated table.")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"Wrote {len(mappings)} validated mappings to {args.output}")


if __name__ == "__main__":
    main()
