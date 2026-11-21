# Arabic OCR for Islamic Manuscripts

Production readiness specification and audit validation for the Arabic OCR pipeline (#214 / #851).

## Architecture

The Arabic OCR service (`arabic_ocr.py`) provides end-to-end processing for historical Islamic manuscripts:
1. **Input validation**: Enforces consent, size caps, and format verification via magic byte sniffing.
2. **Preprocessing**: Profiles for pristine, aged, damaged, faded, and palimpsest manuscripts (deskew, binarization, CLAHE, denoising).
3. **Calligraphy style detection**: Identifies script hands (Naskh, Thuluth, Nastaliq, Maghribi, Kufi, Diwani, Ruqah) to guide recognition parameters.
4. **OCR execution**: Abstracted backend with provider configuration (stub engine for testing, vision providers for production).
5. **Post-processing**: Arabic-specific ligature corrections, hamza normalization, and tashkeel (diacritics) preservation.
6. **Confidence calibration & safe fallback**: Multi-level confidence metrics with automatic review routing.

## Production Contract

### Input Limits & Formats
- **Maximum Image Size**: 10MB (`MAX_IMAGE_BYTES = 10485760`). Payloads exceeding this limit return HTTP 413.
- **Base64 String Cap**: Capped at 15,000,000 characters to prevent memory exhaustion during decoding.
- **Magic Byte Sniffing**: Formats are validated by examining initial header bytes, not MIME headers:
  - PNG: `\x89PNG\r\n\x1a\n`
  - JPEG: `\xff\xd8\xff`
  - TIFF: `II*\x00` (little-endian) or `MM\x00*` (big-endian)
  - WebP: `RIFF....WEBP`
  Unsupported formats return HTTP 415. Empty payloads return HTTP 400.

### Consent & Retention
- **Consent Enforcement**: Mutating requests require `consent_to_process=True`. Requests where consent is false return HTTP 400.
- **Retention Policy**: Strictly `ephemeral_in_memory_only`. Neither raw imagery nor extracted text is persisted to disk, database, or cache. Responses echo `retention_policy: "ephemeral_in_memory_only"`.

## Confidence Scoring & Safe Fallback

### Metric Calibration
The overall confidence score is computed as a weighted combination:
- Character-level confidence: 50%
- Word-level confidence: 30%
- Diacritical mark (tashkeel) confidence: 20%
- Damaged region penalty: Subtractive penalty up to 30% based on detected tears, stains, or ink bleeding.

### Confidence Bands
- **High** (>= 0.70): Reliable automated transcription.
- **Medium** (0.40 - 0.69): Moderate confidence; `review_required = True`.
- **Low** (< 0.40): Sub-threshold confidence; `fallback_recommended = True`, `review_required = True`.

### Safe Fallback Mode
When overall confidence falls below 0.40, damaged region penalty reaches >= 0.25, or no legible Arabic text is detected:
- `fallback_mode` is set to `True`.
- `fallback_reason` provides an actionable diagnostic explanation.
- An explicit warning is appended to `warnings` instructing consumers that the transcription is a tentative draft requiring scholar paleographer verification before publication or citation.

## Configuration & Environment

| Variable | Type | Default | Description |
| --- | --- | --- | --- |
| `ARABIC_OCR_PROVIDER` | string | `stub` | OCR provider backend (`stub`, `google_vision`, `tesseract`) |
| `ARABIC_OCR_MAX_IMAGE_BYTES` | integer | `10485760` | Maximum allowed image upload size in bytes (10MB) |
| `ARABIC_OCR_MIN_CONFIDENCE` | float | `0.40` | Threshold below which safe fallback mode engages |

## Latency & Cost Profile

- **Stub Provider (Local / Dev / CI)**: 5 - 15ms per request; zero monetary cost, zero external network calls.
- **Vision Cloud Provider**: 600 - 1500ms typical latency depending on page resolution; upstream cost bounded by 10MB payload limit.

## Deployment & Rollback

- **Stateless Operation**: The module introduces no database migrations or persistent state.
- **Rollback**: Safe to roll back immediately to the previous deployment artifact. External clients will observe standard fallback behavior without data corruption.
