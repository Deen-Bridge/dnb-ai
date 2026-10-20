# Islamic spaced repetition

The review API schedules learner supplied hadith, Arabic vocabulary, fiqh
rulings, historical dates, and terminology. It stores content and per-card
progress durably in SQLite, independently of the generated flashcards returned
by `/study/generate`. Generated cards can be imported by adding `content_type`
to each `{front, back, tags}` object. Content and citations remain the author's
responsibility; importing a card does not authenticate its religious claims.

All routes require the app's `X-API-Key` service credential. The trusted backend
must resolve the signed-in learner and supply their opaque `user_id` (up to 128
characters). The service key is a backend credential; do not expose it to
browsers. The key grants service access, not end-user authentication.

## Create and import

`POST /study/repetition/{user_id}/cards` creates a card and returns HTTP 201:

```json
{
  "front": "What does sabr mean?",
  "back": "Patience and perseverance",
  "content_type": "vocabulary",
  "source": "Class notes, lesson 1",
  "tags": ["arabic"],
  "audio_url": "https://example.org/sabr.mp3",
  "image_url": "https://example.org/sabr.png"
}
```

`content_type` is `hadith`, `vocabulary`, `ruling`, `date`, or `terminology`.
Media references and the source are optional. Audio and image URLs must use
HTTP(S). The service stores references without downloading or executing them;
a learning client can render them alongside the card. No audio/image generation
or uploads are involved.

`POST /study/repetition/{user_id}/import` accepts `{"cards": [...]}` with 1–500
cards in the same format. Validation covers the entire batch before one atomic
transaction inserts it. Each import creates new IDs and fresh schedules; it is
not a progress restore or duplicate detection mechanism.

## Review

1. Fetch `GET /study/repetition/{user_id}/session?limit=20` to get currently due
   cards, ordered by due time and then ID. Limit is 1–200. New cards are due
   immediately. Fetch again after completing the batch to continue a busy day.
2. Display the front and any media, let the learner reveal the back, then ask
   them to grade their recall from 0 (forgotten) to 5 (perfect). Grades 3–5
   count as successful recall.
3. Send `POST /study/repetition/{user_id}/cards/{card_id}/review` with
   `{"grade": 4, "revision": 0}`. Use the revision returned with that card.
   The response contains the updated schedule and revision.
4. A stale/duplicate revision or an early review returns 409 without modifying
   progress. Reload with `GET /study/repetition/{user_id}/cards/{card_id}`.
   Cross-user and unknown IDs return 404.

The scheduler implements [SM-2](https://super-memory.org/archive/english/ol/sm2.htm):
start ease at 2.5, schedule successful recalls at 1 day and 6 days, then multiply
the preceding interval by the preceding ease and round up. After each grade,
update ease using `EF + 0.1 - (5-q)*(0.08 + (5-q)*0.02)`, floored at 1.3.
Intervals are capped at 36,500 days to keep date arithmetic bounded. A failed
recall resets consecutive repetitions and is due in ten minutes for relearning.
This explicit relearning delay is an adaptation of SM-2. `interval_days` on a
failed card is 1, the starting interval of its reset sequence; `due_at` is the
authoritative time for its ten minute retry. Review times come from the server.
All timestamps are timezone aware UTC; one day means 24 hours.

## Statistics and persistence

`GET /study/repetition/{user_id}/statistics` returns card counts, currently due
and reviewed counts, review/recall/lapse totals, counts by content type, and
`retention_rate`. Retention is the observed ratio of grades 3–5 to all graded
reviews, including relearning attempts. It is `null` before the first review.
Each card also exposes this ratio, ease, consecutive repetitions, intervals,
lapses, and last/next review times. This is a measured recall statistic, not a
calibrated prediction of future recall or a claim about a learner's knowledge.

Set `REPETITION_DB_PATH` (default `data/repetition.db`) to a path on persistent
storage. The database and SQLite WAL/SHM files are ignored by Git. Transactions
and revision checks prevent lost updates or duplicate grading across worker
connections. Mount persistent storage and back up the database for production;
ephemeral deployments lose their local files. SQLite is appropriate for workers
sharing one local volume; separate hosts must share a database service before
using this API across replicas. No external AI provider is needed for reviews.

Offline verification: `pytest -q tests/test_spaced_repetition.py`.
