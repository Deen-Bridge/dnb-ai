# Recitation result calibration and scholar review boundary (AI-081, #856)

Owner: Multimodal (`quran_recitation.py`, `POST /quran-recitation/analyze`).

## What the endpoint judges, and what it does not

The analyser aligns an ASR **transcript** with the bundled Hafs Mushaf text. Its
`verification` and `accuracy` describe how well the text matches, and nothing
else. It never hears audio, so it cannot rule on tajwid. Every response now says
so in `scholar_review`:

| Field | Meaning |
| --- | --- |
| `assessed` | `surah_and_ayah_identification`, `word_level_text_match_against_bundled_hafs_mushaf`, `riwayah_textual_variants_listed_in_module` |
| `not_assessed` | `tajwid_rules`, `waqf_and_ibtida`, `pronunciation_and_articulation`, `reciter_identity_from_voice`, `maqam_from_pitch`, `ijazah_or_certification` |
| `boundary` | Fixed statement: this is an automated text-alignment result, it cannot certify a recitation, and a qualified teacher must confirm any correction |
| `required` | Equal to the top-level `review_required` |
| `reasons` | Machine-readable codes, listed below |

### Review reasons (in reporting order)

| Code | Raised when |
| --- | --- |
| `unidentified` | No bundled passage matched. Informational only; `required` is `false` because there is no result to review |
| `asr_confidence_unknown` | The caller did not send `asr_confidence` |
| `asr_confidence_below_high` | `asr_confidence` < 0.75 |
| `text_deviations` | `verification` is not `accurate` (omission, substitution or insertion) |
| `non_hafs_reading` | A textual variant indicates a riwayah other than Hafs/Shu'bah, while the bundled Mushaf follows Hafs |
| `riwayah_hint_conflict` | The caller's `riwayah_hint` contradicts the textual evidence (also exposed as `riwayah.hint_conflict`) |

## Calibration

```
calibrated_confidence = clamp(accuracy × asr_factor, 0, 1)
asr_factor            = asr_confidence, or UNKNOWN_ASR_PRIOR (0.70) when not supplied
```

`confidence_band`, `fallback_recommended` and the band's review flag come from
`media_upload_limits.evaluate_confidence_and_fallback`, the same thresholds
(`high` ≥ 0.75, `medium` ≥ 0.40, `low` below) that the OCR and video endpoints use.
`review_required` is true if the band asks for review or any review reason applies.

The prior sits below the `high` threshold on purpose: a transcript with
unmeasured ASR quality can never be reported as high confidence. This matches
how `media_upload_limits` treats a missing confidence (`medium`, review required).

Each response carries `calibration_version` (`recitation-calibration/v1`). Bump
it whenever the formula, the prior, the thresholds or the reasons change.

### Behaviour changes

| Situation | Before | After |
| --- | --- | --- |
| No `asr_confidence` sent | `high`, no review | `medium`, review (`asr_confidence_unknown`) |
| Unidentified or empty transcript | `high`, no fallback | `low`, `fallback_recommended: true` |
| `inaccurate` text with ASR ≥ 0.75 | `high`, no review | band from the calibrated score, review (`text_deviations`) |
| Non-Hafs reading or riwayah hint conflict | note text only | review with a structured reason |

Clients that only read `identified`, `verification` and `accuracy` are unaffected.
All new fields are additive.

## Safe fallback

When `fallback_recommended` is true, clients must not present the alignment as a
verdict on the recitation. They should show the `note` and direct the learner to
a teacher. When `review_required` is true, any "correction" shown to the learner
must be labelled as unconfirmed until a qualified reviewer checks it.

## Input limits, consent, retention, formats

These are unchanged and documented in `docs/MULTIMODAL_UPLOAD_LIMITS.md`:

- transcript ≤ 20,000 characters, ≤ 5,000 word timings, duration ≤ 3,600 s
- `asr_confidence` must be in [0, 1] (otherwise 422)
- `consent_to_process: false` returns 400
- input is text only (an ASR transcript); there is no audio upload on this endpoint
- nothing is stored: requests are processed in memory and discarded

## Cost and latency (measured)

`python -m scripts.eval_recitation_calibration` runs 12 labelled cases. The
covered cases are clean, unknown ASR, medium ASR and low ASR, omission, heavy
substitution, Hafs and non-Hafs readings, a riwayah conflict, unrelated Arabic,
and English. While it runs, every socket connection is blocked.

Measured locally (Windows, Python 3.11.9, 20 timed calls per case):

```
recitation-calibration/v1: 12/12 cases match labels; max p95 latency 12.996 ms; provider calls 0; estimated cost $0.00
```

The analyser is deterministic and makes no provider or network calls, so its
marginal cost is zero. Any upstream ASR is the caller's cost. CI checks the same
cases with a 250 ms p95 ceiling.

## Observability

No new logs or metrics. The response itself is the audit record:
`calibration_version`, `calibrated_confidence` and `scholar_review.reasons` can be
counted by the API gateway. No transcript text needs to be logged.

## Rollback

Revert the commit. Response fields are only added, never removed or renamed, so
clients that parse the new fields must handle them being absent after a
rollback. Configuration and data are unchanged.

## Reproducing

```bash
pytest -q tests/test_recitation_calibration.py tests/test_quran_recitation.py
python -m scripts.eval_recitation_calibration
```

Both run offline and need no secrets.
