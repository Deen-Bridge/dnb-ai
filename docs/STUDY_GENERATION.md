# Study Content Generation and Learner Personalization

Production readiness specification and audit validation for deterministic study generation and opt-in resettable personalization (#846 / #847).

## Architecture

The study generation subsystem (`study.py`) creates quizzes and flashcards for authentic Islamic education:
1. **Schema validation**: Translates Pydantic models to strict Gemini JSON schemas (stripping titles and inlining definitions).
2. **Deterministic execution**: When an explicit `seed` parameter is supplied, generation uses an authentic canonical curriculum bank (`DeterministicCurriculumGenerator`) with reproducible shuffling, guaranteeing bit-for-bit identical questions, choices, answers, and explanations.
3. **Provenance & audit trail**: Output envelopes carry versioned curriculum metadata, SHA-256 source text hashing, authentic citations (Quran and Hadith), uncertainty bounds, and timestamps.
4. **Adaptive knowledge assessment**: Integrates discipline-level diagnostics (`knowledge_assessment.py`) across five core Islamic sciences (Aqeedah, Fiqh, Hadith, Tafsir, Seerah).
5. **Opt-in, resettable personalization**: Learner mastery and history are isolated per user in `study_store.py` (in-memory LRU cache and Redis), strictly requiring explicit opt-in (`opt_in_personalization=True`), and completely erasable upon request.

## Production Contract

### Deterministic Generation & Grounding
- **Seed-Driven Reproducibility**: Providing integer `seed` ensures reproducible output across test, audit, and production environments.
- **Canonical Authentic Bank**: Built-in curriculum bank contains verified multi-choice questions and flashcards referencing primary sources:
  - Aqeedah: Surah Adh-Dhariyat 51:56, Hadith of Jibril (Sahih Muslim 8)
  - Fiqh: Surah Al-Baqarah 2:144, Nisab of Gold (Sunan Abi Dawud 1573)
  - Hadith: Mutawatir transmission definitions (Ibn al-Salah, Ibn Hajar)
  - Tafsir: Ayat al-Kursi (Sahih Muslim 810), Al-Kawthar (Sahih al-Bukhari 4964)
  - Seerah: Fath Makkah (Sahih al-Bukhari 4275), Bay'at al-Ridwan (Quran 48:18)
- **Provenance Envelope**:
  - `curriculum_version`: Semantic version of syllabus rules (`1.0.0`).
  - `source_hash`: SHA-256 hex digest of the input topic or lesson text.
  - `citations`: De-duplicated list of authentic scripture and collection references.
- **Audit Trail**:
  - `deterministic`: Boolean indicating whether generation used a deterministic seed.
  - `uncertainty_score`: Metric indicating potential model deviation (<= 0.05 for grounded curricula).
  - `limitations`: Explicit advisory stating content is for educational review only.

### Privacy-by-Default Opt-In Personalization
- **Opt-in Contract**: By default, `opt_in_personalization=False`. When false, passing `user_id` does NOT record study records or track mastery, guaranteeing privacy by default.
- **Opt-in Tracking**: When `opt_in_personalization=True`, progress records are maintained under `study:user:{user_id}`.
- **Isolated Storage**: Stored records contain only aggregate counts, mastery scores, and topics studied (no raw conversational text).
- **Idempotent Reset**: Calling `DELETE /study/users/{user_id}` or `DELETE /knowledge-assessment/users/{learner_id}` permanently and immediately erases all stored personalization data from both cache and Redis.

## API Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/study/generate` | Generate quizzes and/or flashcards with optional seed and opt-in tracking |
| `GET` | `/study/users/{user_id}/progress` | Retrieve study mastery and completed quiz statistics for opted-in user (404 if not found) |
| `DELETE` | `/study/users/{user_id}` | Permanently erase all study progress and personalization data for user (idempotent) |
| `GET` | `/knowledge-assessment/questionnaire` | Retrieve diagnostic questionnaire across core disciplines |
| `POST` | `/knowledge-assessment/evaluate` | Evaluate learner answers, calculate knowledge growth delta, and build knowledge map |
| `DELETE` | `/knowledge-assessment/users/{learner_id}` | Permanently erase learner diagnostic assessment history |

## Configuration & Deployment

- **In-Memory Store**: Default for local development and CI; capped at 5000 users with LRU eviction.
- **Redis Store**: Activated automatically when `REDIS_URL` is set, storing user records with a 1-year rolling TTL (`STUDY_PROGRESS_TTL_SECONDS = 31536000`).
- **Rollback**: Stateless API behavior. Reverting deployment artifacts does not corrupt existing Redis keys (`schema_version` compatible).
