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
* **Label, don't mutate.** Flags never change the text. A lecture that quotes
  "you are now..." or a manuscript that mentions a "system" must survive
  extraction intact. (Open PR #892, for AI-075, proposes removing
  instruction-like spans from `/audio-hadith/verify` input before corpus
  matching. That is a narration-specific safeguard and merges cleanly with this
  boundary.)
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

| Route | Limit | Default | Over the limit |
| --- | --- | --- | --- |
| `POST /video-analysis/analyze` | upload size (`VIDEO_MAX_BYTES`), streamed to disk | 100 MB | `413 payload_too_large` |
| | empty upload | | `422 empty_upload` |
| | extension | `.mp4 .mov .avi .webm` | `415 unsupported_format` |
| | `transcript` form field (`VIDEO_MAX_TRANSCRIPT_CHARS`) | 200,000 chars | `413 transcript_too_large` |
| | queued + running jobs (`VIDEO_MAX_PENDING_JOBS`) | 8 | `503 service_busy` |
| | concurrent processing (`VIDEO_MAX_CONCURRENT_JOBS`) | 2 | later jobs wait in the queue |
| | media duration processed (`VIDEO_MAX_DURATION_SECONDS`) | 7,200 s | ffmpeg `-t` stops there |
| | frames OCR'd (`VIDEO_MAX_FRAMES`) | 240 (1 per 30 s) | ffmpeg `-frames:v` stops there |
| | each ffmpeg call (`VIDEO_FFMPEG_TIMEOUT_SECONDS`) | 120 s | job `processing_timeout` |
| | extracted text returned (`VIDEO_MAX_EXTRACTED_CHARS`) | 200,000 chars | cut, flag `truncated` |
| | topics returned | 500 | cut |
| `POST /page-analysis/analyze` | blocks per page | 2,000 | `422` |
| | text per block | 10,000 chars | `422` |
| | text per page | 200,000 chars | `422` |
| | geometry | finite numbers | `422` for `inf` / `NaN` |
| `POST /audio-hadith/lecture-transcript` | `video_ref` | 200,000 chars (unchanged) | `422` |
| `/audio-hadith/*` | `language` | 16 chars (lecture: language-tag pattern) | `422` |

## 3. Retention and cleanup

* The uploaded video is written to a private temporary file and deleted when
  its job ends, whether it succeeds, fails or times out. A rejected upload is
  deleted immediately.
* Extracted audio and frames live in a temporary directory removed when
  analysis returns or raises.
* Job records, including transcripts, are kept **in process memory only**,
  for `VIDEO_JOB_RETENTION_SECONDS` (default 3,600) after completion and at
  most `VIDEO_MAX_STORED_JOBS` (default 200) jobs; the oldest finished jobs are
  evicted first and active jobs never are. Nothing is written to a database,
  and everything is lost on restart. Finished jobs report `expires_at`.
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
Exception text, ffmpeg output, file paths and media content are never returned.
Upload rejections return the service's usual error envelope with an added
stable `code`, for example `{"code": "unsupported_format", "detail": "Only MP4, MOV, AVI, and WebM videos are supported."}`.

## 5. Observability

* One INFO line per finished video job: job id, status, `error_code`, input
  bytes and `metrics` (frames processed, transcript and OCR character counts,
  processing milliseconds, provider cost). No transcript or OCR text is
  logged; the test suite asserts this.
* Unexpected failures are logged at ERROR with the exception **type and stack
  frames only**, not the exception message (which could quote media content).
* Useful alerts: rate of `503 service_busy` (raise `VIDEO_MAX_PENDING_JOBS` or
  add workers), `processing_timeout` and `decode_failed` job rates, and the
  share of results with `content_trust.instruction_like: true`.

## 6. Cost and latency evidence

These routes make no paid provider calls, so provider cost is $0 per request.
Media decoding, ASR and OCR run on the worker CPU and are bounded by the
duration and frame caps. `scripts/benchmark_extraction_limits.py` measures this
service's own processing at representative and worst allowed sizes (fakes
replace ffmpeg/ASR/OCR):

| Surface | Input | Best time (ms) | Output |
| --- | --- | --- | --- |
| page-analysis | 60 blocks, 6k chars | 2.8 | 60 elements |
| page-analysis | worst: 2,000 blocks, 200k chars | 84.9 | 2,000 elements |
| lecture-transcript | 20k chars | 11.0 | 267 segments |
| lecture-transcript | worst: 200k chars | 95.8 | 2,667 segments |
| video-analysis (no media) | 20k-char transcript | 6.7 | 267 topics |
| video-analysis (no media) | worst: 200k-char transcript | 66.2 | 500 topics (capped) |
| trust inspection | worst: 200k chars of near-miss patterns | 81.2 | 0 flags |

Measured with Python 3.11.18 on Linux x86_64, best of 5 runs. Re-run the script
on the target machine for numbers that apply there.

## 7. Configuration

All variables are optional; defaults are in the limits table. Invalid values
fall back to the default with a warning. See `.env.example`.

## 8. Deployment and rollback

* **Compatibility.** Additive fields: `content_trust` everywhere;
  `extraction_status`, `reference_status` on page analysis; `summary_method`,
  `reference_status`, `extraction_status`, `metrics`,
  `capabilities.media_decoded` on video results; `error_code`, `expires_at` and
  `input_bytes` on jobs. Changed behaviour that clients may notice:
  * upload rejections add a `code` field (`detail` is still a string);
  * a failed job's `error` is a fixed safe message instead of the raw exception;
  * results disappear after the retention window (`404` on poll);
  * new `413`/`422`/`503` rejections for inputs beyond the limits above;
  * role-like lecture prefixes (`System:`, `User:`, ...) are no longer
    speaker labels.

  Before deploy, check access logs for page-analysis requests above 2,000
  blocks or 200k characters and for video transcripts above 200k characters;
  after deploy, watch the `413`/`422`/`503` rates on these routes.
* **Rollback.** Redeploy the previous image. No storage, schema or data
  migration is involved; in-memory jobs are lost on any restart either way.
  Limits can also be relaxed without a rollback through the environment
  variables above.
