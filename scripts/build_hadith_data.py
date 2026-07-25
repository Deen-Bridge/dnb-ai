"""Builds data/hadith/*.json — the bundled hadith grading dataset.

Source: fawazahmed0/hadith-api (https://github.com/fawazahmed0/hadith-api),
pinned to tag "1", released under the Unlicense (public domain). We only
extract grading metadata (collection, book/hadith numbers, grader names,
grade strings) — never the translated hadith text — so no translation
copyright question ever arises.

Re-run this script to refresh the bundled dataset:

    python scripts/build_hadith_data.py

It requires network access and is not run as part of CI; CI tests run
entirely offline against the committed data/hadith/*.json files.

See data/hadith/PROVENANCE.md for full provenance and the grading
normalization policy (implemented in hadith.py, shared by this script and
the runtime lookup so there is exactly one normalizer).
"""

import json
import sys
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import hadith  # noqa: E402

SOURCE_REPO = "fawazahmed0/hadith-api"
SOURCE_TAG = "1"
BASE_URL = f"https://cdn.jsdelivr.net/gh/{SOURCE_REPO}@{SOURCE_TAG}/editions"

# Bukhari and Muslim carry no per-hadith grade in the source: classical
# scholarship treats both as authenticated by consensus (ijma') in full.
CONSENSUS_COLLECTIONS = {"bukhari", "muslim"}
CONSENSUS_GRADER = "Scholarly consensus (Bukhari and Muslim)"

COLLECTIONS = ["bukhari", "muslim", "abudawud", "tirmidhi", "nasai", "ibnmajah", "malik"]

OUT_DIR = ROOT / "data" / "hadith"


def fetch_edition(collection: str) -> dict:
    url = f"{BASE_URL}/eng-{collection}.min.json"
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def build_record(raw_hadith: dict, collection: str) -> dict:
    reference = raw_hadith.get("reference", {})
    raw_grades = raw_hadith.get("grades") or []

    graders = []
    strengths = []
    chain_types = []

    if collection in CONSENSUS_COLLECTIONS:
        strength, chain_type = hadith.Strength.SAHIH, hadith.ChainType.MARFU
        graders.append({"g": CONSENSUS_GRADER, "raw": "Sahih (ijma')", "s": strength.value, "c": chain_type.value})
        strengths.append(strength)
        chain_types.append(chain_type)
    else:
        for entry in raw_grades:
            strength, chain_type = hadith.parse_grade_string(entry.get("grade", ""))
            graders.append(
                {
                    "g": entry.get("name", "Unknown"),
                    "raw": entry.get("grade", ""),
                    "s": strength.value,
                    "c": chain_type.value,
                }
            )
            strengths.append(strength)
            chain_types.append(chain_type)

    return {
        "n": raw_hadith["hadithnumber"],
        "book": reference.get("book"),
        "bn": reference.get("hadith"),
        "an": raw_hadith.get("arabicnumber"),
        "grade": hadith.aggregate_strength(strengths).value,
        "chain": hadith.aggregate_chain_type(chain_types).value,
        "graders": graders,
    }


def build_collection(collection: str) -> None:
    print(f"Fetching {collection}...")
    edition = fetch_edition(collection)
    records = [build_record(h, collection) for h in edition["hadiths"]]

    payload = {
        "collection": collection,
        "name": edition.get("metadata", {}).get("name", collection),
        "source": {"repo": SOURCE_REPO, "ref": SOURCE_TAG, "fetched": date.today().isoformat()},
        "hadiths": records,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{collection}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    print(f"  wrote {len(records)} hadiths to {out_path}")


def main() -> None:
    for collection in COLLECTIONS:
        build_collection(collection)


if __name__ == "__main__":
    main()
