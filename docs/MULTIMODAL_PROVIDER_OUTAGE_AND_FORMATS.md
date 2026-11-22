# Multimodal Provider Outage and Unsupported Format Specification

## 1. Production Contract and Architecture

This document specifies the production boundaries, failure modes, error handling, and security controls for multimodal processing services in `dnb-ai`:
- **Video Analysis** (`/video-analysis`): Lecture processing, chapter segmentation, transcription, visual OCR, and reference extraction.
- **Arabic OCR** (`/arabic-ocr`): Manuscript page processing, calligraphy classification, diacritic preservation, and text extraction.
- **Quran Recitation** (`/quran-recitation`): Audio/transcript alignment against the bundled Mushaf with calibrated multi-dimensional confidence.

## 2. Untrusted Content Isolation and Prompt Injection Defenses

Extracted text from audio, video frames, or manuscript images is untrusted user-supplied data:
- **Defense Mechanism**:
  - All extracted text is sanitized by `untrusted_content.sanitize_untrusted_text()`.
  - Zero-width spaces and Unicode bidirectional overrides (`\u200B` - `\u200D`, `\uFEFF`, `\u202A` - `\u202E`) are stripped.
  - Prompt injection sequences (e.g. `[SYSTEM]`, `system:`, `<|im_start|>`, `ignore previous instructions`) are defanged and redacted.
  - Text returned to downstream consumers is packaged with `wrap_untrusted_content()` containing explicit structural delimiters (`<<<UNTRUSTED_EXTRACTED_DATA>>>`) and `is_untrusted: true` flags. Downstream LLM orchestrators treat this data purely as payload, preventing instruction hijacking.

## 3. Provider Outages and Graceful Degradation

### Video Analysis
- **FFmpeg Unavailable or Extraction Failure**: Falls back to transcript-only analysis when text is provided; returns empty frame array without crashing.
- **ASR (Faster-Whisper) Outage**: If local ASR packages fail to load or time out, processing degrades to visual OCR frames and metadata extraction.
- **Job Failure**: Unhandled processing crashes mark the job as `failed` with sanitized error details (`PROVIDER_PROCESSING_ERROR`), ensuring internal file paths and stack traces are not leaked.

### Arabic OCR
- **External OCR Engine Outage**: When calling external providers (Google Cloud Vision, Azure Cognitive Services, or custom endpoints), network errors or HTTP 5xx responses produce an explicit HTTP 503 (`OCR engine provider outage or unavailable`).
- **Engine Timeouts**: Custom endpoints enforce a strict 10.0-second timeout to prevent thread starvation.
- **Damaged Manuscript Penalty**: If image processing detects significant damage, confidence metrics apply a penalty rather than raising errors.

### Quran Recitation
- **Transcriber Outage**: The pluggable `Transcriber` interface isolates upstream ASR. If no transcript can be produced, the endpoint returns an explicit HTTP 422 or abstains safely with `identified: false`.

## 4. Unsupported and Malformed Formats

Uploads are validated before processing via `media_upload_limits.py`:
- **Video**: Accepts MP4, MOV, AVI, WebM up to 100MB and 1800s. Corrupted headers or non-matching magic bytes return HTTP 415 (`Unsupported video format or corrupted file header`).
- **Images / Manuscripts**: Accepts PNG, JPEG, TIFF, WebP up to 10MB. Invalid base64 or unsupported image magic bytes return HTTP 415 / 422.
- **Audio / Transcripts**: Max duration 3600s, max transcript length 20,000 chars, max word timings 5,000. Excess lengths return HTTP 413 or 422.
- **Consent Requirement**: All processing endpoints require `consent_to_process=true`. Requests missing consent return HTTP 400.

## 5. Deployment and Rollback Procedures

### Deployment
1. Verify required environment variables in Render/Docker:
   - `GEMINI_API_KEY`: API key for generative fallback.
   - `SERVICE_API_KEY`: Internal service authentication.
   - `REDIS_URL`: Persistent Redis connection for cache and jobs.
   - `ENVIRONMENT=production`: Enforces strict durability mode.
   - `MODERATION_STRICT_DURABILITY=true`: Prevents ephemeral moderation writes.
   - `MODERATION_FALLBACK_FILE=/var/data/moderation_store.json`: Persistent disk path.
2. Deploy container image via Render or Docker compose.
3. Validate health check endpoint `GET /health` returns status `ok`.

### Rollback
1. If provider outage or unexpected failures occur post-deployment:
   - Revert container image to previous release SHA.
   - The durable fallback file at `/var/data/moderation_store.json` preserves moderation records across rollbacks.
   - Trigger `POST /books/moderation/reconcile` with admin credentials to reconcile caches.
