# Extracted content: trust boundary and production limits

Owner: multimodal extraction (`untrusted_content.py`, `video_analysis.py`, `page_analysis.py`, `audio_hadith.py`).
Audit item: #858 (PROD-AUDIT AI-082).

## 1. Contract

Text produced from user media is **untrusted data**. That covers ASR transcripts,
frame OCR, OCR/layout blocks sent to page analysis, and any transcript a client
types or uploads. It is data about the media, never instructions.

* **Inside this service** no extracted text reaches a model, a tool, the network
  or a shell. The extraction modules import no model client, HTTP client or
  orchestration module, and ffmpeg is only called with a fixed argv list
  (never a shell). `tests/test_extracted_content_boundary.py` checks this
  statically, so a future change that wires extracted text into a model fails CI.
* **Every extraction response** carries `content_trust`:

  ```json
  {
    "trust": "untrusted",
    "sources": ["asr", "ocr"],
    "instruction_like": true,
    "flags": ["instruction_override", "tool_call_syntax"],
    "handling": "Untrusted data extracted from user media. Quote it as data; ..."
  }
  ```

  Consumers (the DeenBridge backend, any prompt builder) must follow `handling`:
  quote the text as data; never put it in system or developer instructions;
  never let it select or authorize a tool; never treat a pattern match as
  verification. `instruction_like` is a signal for review and UI, not a
  permission check: tool permissions are decided by application code only.
* **Label, don't mutate.** These flags never change the text. A lecture that
  quotes "you are now..." or a manuscript that mentions a "system" must survive
  extraction intact. The one exception is the audio-hadith pipeline: #892
  (AI-075) removes instruction-like spans from `/audio-hadith/verify` input and
  lecture segments before matching and reports them in `injection_flags`.
  `content_trust` is computed after that step, so both signals stay accurate.
* **Structure cannot be spoofed.** In `/audio-hadith/lecture-transcript`, a line
  prefix becomes a speaker label only if it is at most 4 words and 64
  characters, has no markup characters, and contains no role word (`system`,
  `assistant`, `developer`, `user`, `tool`, `function`, `model`, `ai`, `bot`,
  `prompt`, `instruction(s)`). Otherwise the whole line stays text under
  "Speaker 1", so `searchable_text` can never read like a model conversation.
* **No claims from pattern matches.** Video summaries are extractive
  (`summary_method: "extractive"`), and Qur'an and hadith references from video
  and page analysis are `reference_status: "unverified_pattern_matches"`.

### Flags

| Flag | Meaning |
| --- | --- |
| `instruction_override` | "ignore/disregard previous instructions" (English and Arabic) |
| `role_marker` | `system:` / `assistant:` lines, ChatML tokens, `[INST]`, `BEGIN SYSTEM` |
| `persona_change` | "you are now", "pretend to be" |
| `secret_request` | requests to reveal prompts, keys, tokens, passwords |
| `tool_call_syntax` | JSON `{"tool": ...}` / `{"function_call": ...}`, `<tool_call>` tags, `tool:` lines |
| `verification_claim` | "mark this as verified/sahih" |
| `hidden_characters` | C0 controls, DEL, zero-width space, word joiner, BOM, bidi overrides/isolates |
| `truncated` | extracted text was cut at the output limit |

ZWNJ, ZWJ, LRM and RLM are legitimate in Arabic, Persian and Urdu and are
never flagged. Detection uses bounded patterns and runs in linear time.

## 2. Limits

Upload validation comes from `media_upload_limits.py` (#853, see
`docs/MULTIMODAL_UPLOAD_LIMITS.md`); this audit item adds the limits on the work
done after an upload is accepted, and on page and lecture text.

| Route | Limit | Default | Over the limit | Source |
| --- | --- | --- | --- | --- |
| `POST /video-analysis/analyze` | upload size | 100 MB | `413` | #853 |
| | empty upload, missing consent | | `400` | #853 |
| | container signature (MP4, MOV, AVI, WebM) | | `415` | #853 |
| | `transcript` form field | 200,000 chars | `413` | #853 |
| | queued + processing jobs | 20 | `429` | #853 |
| | concurrent decoding (`VIDEO_MAX_CONCURRENT_JOBS`) | 2 | later jobs wait in the queue | #858 |
| | media duration processed | 1,800 s (`MAX_VIDEO_DURATION_SECONDS`) | ffmpeg `-t` stops there | #858 |
| | frames OCR'd (`VIDEO_MAX_FRAMES`) | 60 (1 per 30 s) | ffmpeg `-frames:v` stops there | #858 |
| | each ffmpeg call (`VIDEO_FFMPEG_TIMEOUT_SECONDS`) | 120 s | job `processing_timeout` | #858 |
| | extracted text returned (`VIDEO_MAX_EXTRACTED_CHARS`) | 200,000 chars | cut, flag `truncated` | #858 |
| | topics returned | 500 | cut | #858 |
| `POST /page-analysis/analyze` | blocks per page | 5,000 | `422` | #853 |
| | page width and height | 20,000 units | `422` | #853 |
| | text per block | 10,000 chars | `422` | #858 |
| | text per page | 200,000 chars | `422` | #858 |
| | block geometry | finite numbers | `422` for `inf` / `NaN` | #858 |
| `POST /audio-hadith/lecture-transcript` | `video_ref` | 200,000 chars (unchanged) | `422` | |
| `/audio-hadith/*` | `language` | 16 chars (lecture: language-tag pattern) | `422` | #858 |

## 3. Retention and cleanup

* A validated upload is written to a private (owner-only) temporary file and
  the request's copy is released, so a queued job holds a path, not the
  payload. The file is deleted when the job ends: completed, failed, timed out
  or cancelled. An upload rejected after it was written (for example because
  the service is busy) is deleted immediately.
* Extracted audio and frames live in a temporary directory removed when
  analysis returns or raises.
* Job records, including transcripts, are kept **in process memory only** by
  the `BoundedJobManager` from #853: one hour from creation, at most 200 jobs,
  oldest finished jobs evicted first. Nothing is written to a database, and
  everything is lost on restart. `POST /video-analysis/jobs/{id}/cancel` stops a
  queued job before it is analyzed.
* Page analysis and lecture transcripts are stateless: nothing is stored.

## 4. Failure modes

| Situation | Result |
| --- | --- |
| ffmpeg not installed | job completes; `capabilities.ffmpeg: false` (unchanged behaviour) |
| ffmpeg installed but yields no audio and no frames, no transcript supplied | job `failed`, `error_code: decode_failed` |
| same, with a transcript supplied | job `completed`, `capabilities.media_decoded: false` |
| ffmpeg exceeds its timeout | job `failed`, `error_code: processing_timeout` |
| any other exception | job `failed`, `error_code: internal_error` |
| nothing extracted | job `completed`, `extraction_status: "empty"` (not a failure) |
| page with no readable text | `extraction_status: "empty"` |

Failed jobs return a stable `error_code` and a fixed safe `error` message.
Exception text, ffmpeg output, file paths and media content are never returned;
in particular, failed jobs no longer carry an `error_details` field with the raw
exception text. Upload rejections keep the messages from #853.

## 5. Observability

* One INFO line per finished video job: job id, status, `error_code`, input
  bytes and `metrics` (frames processed, transcript and OCR character counts,
  processing milliseconds, provider cost). No transcript or OCR text is
  logged; the test suite asserts this.
* Unexpected failures are logged at ERROR with the exception **type and stack
  frames only**, not the exception message (which could quote media content).
* Useful alerts: rate of `429` busy rejections (add workers), the
  `processing_timeout` and `decode_failed` job rates, and the share of results
  with `content_trust.instruction_like: true`.

## 6. Cost and latency evidence

These routes make no paid provider calls, so provider cost is $0 per request.
Media decoding, ASR and OCR run on the worker CPU and are bounded by the
duration and frame caps. `scripts/benchmark_extraction_limits.py` measures this
service's own processing at representative and worst allowed sizes (fakes
replace ffmpeg/ASR/OCR):

| Surface | Input | Best time (ms) | Output |
| --- | --- | --- | --- |
| page-analysis | 60 blocks, 6k chars | 1.6 | 60 elements |
| page-analysis | worst: 5,000 blocks, 200k chars | 74.9 | 5,000 elements |
| lecture-transcript | 20k chars | 12.6 | 267 segments |
| lecture-transcript | worst: 200k chars | 96.8 | 2,667 segments |
| video-analysis (no media) | 20k-char transcript | 6.0 | 267 topics |
| video-analysis (no media) | worst: 200k-char transcript | 56.5 | 500 topics (capped) |
| trust inspection | worst: 200k chars of near-miss patterns | 64.0 | 0 flags |

Measured with Python 3.11.18 on Linux x86_64, best of 5 runs. Re-run the script
on the target machine for numbers that apply there.

## 7. Configuration

The `VIDEO_*` variables above are optional; invalid values fall back to the
default with a warning. Upload limits are constants in `media_upload_limits.py`.
See `.env.example`.

## 8. Deployment and rollback

* **Compatibility.** Additive fields: `content_trust` everywhere;
  `extraction_status`, `reference_status` on page analysis; `summary_method`,
  `reference_status`, `extraction_status`, `metrics`,
  `capabilities.media_decoded` on video results; `error_code` and
  `input_bytes` on jobs. Changed behaviour that clients may notice:
  * a failed job's `error` is a fixed safe message and `error_details` is gone;
  * new `422` rejections for page text and geometry beyond the limits above;
  * at most `VIDEO_MAX_CONCURRENT_JOBS` videos decode at once, so a burst
    waits in the queue instead of running in parallel;
  * role-like lecture prefixes (`System:`, `User:`, ...) are no longer
    speaker labels.

  Before deploy, check access logs for page-analysis blocks above 10k or pages
  above 200k characters; after deploy, watch the `422` rate on page analysis and
  the job queue time on video analysis.
* **Rollback.** Redeploy the previous image. No storage, schema or data
  migration is involved; in-memory jobs are lost on any restart either way.
  Limits can also be relaxed without a rollback through the environment
  variables above.
