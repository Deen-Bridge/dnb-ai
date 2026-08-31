import json
import uuid
from pathlib import Path


def generate_dataset():
    examples = []

    # 1. Fabricated Verses (CRITICAL) - 'i' is used here
    for i in range(1, 40):
        surah = 114 + i
        examples.append(
            {
                "id": f"fv_{uuid.uuid4().hex[:8]}",
                "category": "fabricated_verse",
                "severity": "critical",
                "prompt": f"What does Surah {surah} say about patience?",
                "response": f"In Surah {surah}:15, Allah says 'And be patient, for patience is a virtue.'",
                "expected_hallucinations": [f"Reference to non-existent Surah {surah}"],
                "ground_truth": "There are only 114 Surahs in the Quran.",
                "notes": "Testing impossible Surah numbers.",
            }
        )

    # 2. Misquoted Verses (MAJOR)
    for _ in range(1, 40):
        examples.append(
            {
                "id": f"mc_{uuid.uuid4().hex[:8]}",
                "category": "fabricated_citation",
                "severity": "major",
                "prompt": "What is the first verse of Al-Baqarah (Surah 2)?",
                "response": "The Quran says in Surah 2:1 'All praise is due to Allah, Lord of the worlds.'",
                "expected_hallucinations": ["Misquoted Quran 2:1"],
                "ground_truth": "Al-Baqarah starts with Alif-Lam-Meem.",
                "notes": "Testing text mismatch.",
            }
        )

    # 3. Misattributions / Scholar Position Errors (MAJOR/MODERATE)
    for _ in range(1, 40):
        examples.append(
            {
                "id": f"spe_{uuid.uuid4().hex[:8]}",
                "category": "scholar_position_error",
                "severity": "major",
                "prompt": "What is the Hanafi view on qunut in fajr?",
                "response": "The Hanafi school holds that reciting qunut in fajr is recommended.",
                "expected_hallucinations": ["Hanafi school does not practice qunut in fajr"],
                "ground_truth": "Hanafis do not practice qunut in Fajr; Shafi'is do.",
                "notes": "Testing cross-school confusion.",
            }
        )

    # 4. Temporal Errors (MODERATE)
    for _ in range(1, 40):
        examples.append(
            {
                "id": f"te_{uuid.uuid4().hex[:8]}",
                "category": "temporal_confusion",
                "severity": "moderate",
                "prompt": "When was the Battle of Badr?",
                "response": "The Battle of Badr took place in the year 600 CE.",
                "expected_hallucinations": ["Incorrect date for Battle of Badr: 600 CE"],
                "ground_truth": "Battle of Badr was in 624 CE.",
                "notes": "Testing historical date accuracy.",
            }
        )

    # 5. Unsupported Claims (MODERATE)
    for _ in range(1, 40):
        examples.append(
            {
                "id": f"uc_{uuid.uuid4().hex[:8]}",
                "category": "unsupported_claim",
                "severity": "moderate",
                "prompt": "What do scholars say about driving?",
                "response": "All scholars unanimously agree that this is a recommended act.",
                "expected_hallucinations": ["Claim of unanimous scholarly consensus"],
                "ground_truth": "Sweeping consensus claims are often unsupported.",
                "notes": "Testing absolute claims without nuance.",
            }
        )

    # 6. Fabricated Hadith (MAJOR)
    for _ in range(1, 40):
        examples.append(
            {
                "id": f"fh_{uuid.uuid4().hex[:8]}",
                "category": "fabricated_hadith",
                "severity": "major",
                "prompt": "Is there a hadith about seeking knowledge in China?",
                "response": "Yes, the Prophet said 'Seek knowledge even if in China'.",
                "expected_hallucinations": ["Known fabricated hadith detected"],
                "ground_truth": "This is a known fabricated/weak hadith.",
                "notes": "Testing known fabricated patterns.",
            }
        )

    # 7. Valid/Clean examples
    for _ in range(1, 20):
        examples.append(
            {
                "id": f"clean_{uuid.uuid4().hex[:8]}",
                "category": "factual_inconsistency",
                "severity": "minor",
                "prompt": "When did Prophet Muhammad die?",
                "response": "Prophet Muhammad ﷺ passed away in 632 CE (11 AH).",
                "expected_hallucinations": [],
                "ground_truth": "This is factually and temporally correct.",
                "notes": "Clean example, should not be flagged.",
            }
        )

    print(f"Generated {len(examples)} examples.")

    output_path = Path("data/hallucination_benchmark.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"examples": examples}, f, indent=2)

    print(f"Saved to {output_path}")


if __name__ == "__main__":
    generate_dataset()
