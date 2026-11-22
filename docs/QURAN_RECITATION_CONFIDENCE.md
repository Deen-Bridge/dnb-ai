# Quran Recitation Matching Confidence Specification

## 1. Overview and Problem Statement

Quranic recitation identification and alignment against the Mushaf requires strict accuracy. Presenting an uncertain, ambiguous, or poorly recorded recitation as confirmed risks misattributing scripture or displaying misleading tajweed corrections.

This specification introduces a multi-dimensional, calibrated confidence model (`v1.0.0`) that guarantees:
1. Clear separation between confirmed matches, review-required candidates, and explicit abstentions.
2. Protection against false-positive match attribution on noisy, truncated, or parallel passages.
3. Machine-readable confidence reporting for frontend rendering and scholarly verification.

## 2. Multi-Dimensional Confidence Model

### Dimensions
The confidence scoring engine evaluates four independent dimensions scored in `[0.0, 1.0]`:

| Dimension | Weight | Definition | Measurement Source |
|---|---|---|---|
| `audio_quality` | 20% | Clarity of the audio signal or ASR acoustic confidence | Upstream ASR confidence or noise estimator |
| `transcription_alignment` | 25% | Fidelity of word alignment against the corpus | Ratio of exact matches to total input tokens |
| `verse_boundary` | 15% | Precision of start and end verse boundaries | Completeness of aligned ayat (full vs partial) |
| `matching_score` | 40% | Combined sequence accuracy and transcript coverage | `accuracy * coverage` against matched surah |

### Calibration Formula
```python
calibrated_confidence = (
    0.20 * audio_quality +
    0.25 * transcription_alignment +
    0.15 * verse_boundary +
    0.40 * matching_score
)
```

## 3. Decision Status and Versioned Thresholds

Version: `CONFIDENCE_MODEL_VERSION = "1.0.0"`

- **CONFIRMED** (`calibrated_confidence >= 0.85` and `accuracy >= 0.85` and not ambiguous):
  - High confidence match.
  - Safe to display confirmed surah name, ayah numbers, and full text in the UI.
- **NEEDS_REVIEW** (`0.50 <= calibrated_confidence < 0.85`, or ambiguous parallel passage):
  - Recitation has partial matches or minor deviations.
  - Advisory only. UI prompts user or scholar to review evidence.
- **ABSTAIN** (`calibrated_confidence < 0.50` or `audio_quality < 0.40`):
  - Audio quality or alignment is insufficient for attribution.
  - Explicit machine-readable explanation provided.
  - Does NOT present uncertain verses or tajweed interpretations as confirmed.

## 4. Frontend Rendering Guidelines

Frontend clients (mobile and web) must display confidence as follows:
- **Confirmed (`CONFIRMED`)**:
  - Render green verification badge: `Verified (Hafs Mushaf)`.
  - Display surah and ayah range with confidence percentage.
- **Needs Review (`NEEDS_REVIEW`)**:
  - Render amber advisory banner: `Suggested Match (Requires Review)`.
  - Highlight ambiguous or approximate words in yellow.
  - Provide a one-click button to view the printed Mushaf passage.
- **Abstain (`ABSTAIN`)**:
  - Render neutral banner: `Recitation Not Confirmed`.
  - Display safe explanatory note. Do not assert verse identity or error attribution.

## 5. Scholar and Reviewer Verification Workflow

Scholars and teachers inspecting flagged recitations receive the `evidence_boundaries` dictionary:
- Surah number and name
- Ayah start and end index
- Total input token count
- Matched word count
- Token-by-token alignment with word-level status (`exact`, `approximate`, `missing`) and timestamps.
