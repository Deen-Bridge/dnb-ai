# Experiment rollout and rollback

An experiment ID acts as a feature flag for a versioned model, prompt, and
generation configuration. Experiments are registered without affecting chat
traffic. An administrator must activate a revision before `/chat` or
`/chat/stream` uses it. With no active experiment, the existing chat path and
streaming behavior remain in use.

## Create and gradually activate a candidate

Send this JSON to `POST /experiments` with the configured `X-Admin-Token`:

```json
{
  "experiment_id": "generation-canary",
  "control_template": "islamic_context",
  "control_version": "1.0.0",
  "variants": [
    {
      "name": "candidate",
      "template_name": "islamic_context",
      "template_version": "1.0.0",
      "model_name": "gemini-2.5-flash",
      "generation_config": {"temperature": 0.2},
      "weight": 1
    }
  ],
  "traffic_percent": 5,
  "rollback_policy": {
    "min_samples": 20,
    "window_size": 100,
    "max_error_rate": 0.1,
    "max_p95_latency_ms": 5000
  }
}
```

The control defaults to the service's configured model and generation settings.
Choose a supported candidate model and a control known to be stable in your
deployment. Prompt names and versions must already exist in the existing
`PromptRegistry`; register new templates through that registry at application
startup. Prompt variants can use `response_language`, `language`, `madhhab`,
`knowledge_level`, `intent`, `retrieved_context`, and `user_question` variables.
Rendered variants retain the chat handler's shared hadith, citation, retrieval,
memory, safety, and confidence processing.

The response includes a monotonically increasing configuration `version`.
Model names, complete generation settings, and prompt bodies/versions are
snapshotted. Later registry changes or process restarts cannot alter a saved
revision. Registering another revision stages it; it does not replace the
currently active revision.

Call `POST /experiments/generation-canary/activate` to activate the newest staged
revision. Alternatively, include `"activate": true` when creating it. Only one
experiment is active for the chat generation path at a time; activating another
one records the prior experiment's deactivation.

`traffic_percent` is the total percentage receiving treatment. The rest receive
the known stable control. Inside the treatment cohort, variant weights choose
between candidates. Assignment hashes the experiment ID and user ID (when
present), otherwise the chat ID, keeping cohorts sticky across requests. Raising
the percentage preserves existing canary membership. A percentage of zero serves
only stable control; 100 includes all sessions in the treatment cohort. Omitting
the percentage preserves the original weighted control/variant A/B split.

To raise a canary from 5% to 25%, register the configuration with
`"traffic_percent": 25`, inspect its new revision, and activate it. Existing
revisions remain available for inspection. A changed control is served only
after an administrator explicitly marks its revision stable.

## Controls and operational views

Mutations and operational detail endpoints require `X-Admin-Token`. They are
closed when `ADMIN_TOKEN` is unset. Invalid configurations return 422; unknown
experiments or versions return 404. Unsafe resume/reactivation attempts after
rollback return 409.

| Endpoint | Effect |
| --- | --- |
| `POST /experiments` | Validate and stage a configuration revision. |
| `GET /experiments` | List current/latest/stable revisions, flags, and configuration metadata. |
| `POST /experiments/{id}/activate` | Activate the newest staged revision. |
| `POST /experiments/{id}/deactivate` | Return new requests to the service's default configuration. |
| `POST /experiments/{id}/kill` | Pause treatment and serve stable control. |
| `POST /experiments/{id}/resume` | Resume a manually paused experiment. |
| `POST /experiments/{id}/rollback` | Immediately restore known stable control. |
| `POST /experiments/{id}/stable` with `{"version": N}` | Explicitly mark revision N's control as the stable baseline. |
| `GET /experiments/{id}/versions` | Inspect immutable configuration and prompt snapshots. |
| `GET /experiments/{id}/metrics` | Inspect per-version/per-variant attempt windows. |
| `GET /experiments/{id}/events` | Inspect the durable state-change audit trail. |
| `DELETE /experiments/{id}` | Remove the experiment from service; retain revision/audit history. |

After evaluating a treatment, register a revision whose control fields match
the verified configuration and mark that revision stable. Marking stable changes
what untreated users and subsequent rollbacks receive, so it is an explicit
operator action. Older stable snapshots remain in version history and can be
selected again through the same stable-version operation.

## Metrics and automatic rollback

Actual generation attempts automatically record latency and success/failure,
tagged with experiment, artifact revision, and variant. Failed and empty
candidate responses count as errors. A failed candidate attempt immediately
retries stable control for that request, without waiting for the aggregate
rollback threshold. Candidate failures do not receive the default model's
transient retries; stable attempts retain the existing retry behavior.

Each variant keeps at most `window_size` observations per revision. Metrics
report `window_attempts`, `errors`, `error_rate`, and nearest-rank
`p95_latency_ms`. Windows measure model attempts, including retry time, rather
than cache hits or unrelated request validation failures. Controls are measured
but cannot trigger a rollback loop. No user IDs, prompts, or answers are stored
in experiment metrics or state-change events.

Once `min_samples` observations exist, an error rate or p95 strictly above its
configured limit rolls the experiment back atomically. Either limit can be
disabled with `null`. Minimum samples must fit inside the observation window;
weights and limits must be finite and valid. Late results from an older revision
cannot roll back a newer active revision.
Requests already running retain their pinned artifacts; subsequent requests
observe the new control-plane decision.

Rollback stops treatment and selects the previous stable model, prompt, and
generation settings. Repeated failures from in-flight requests produce only one
automatic rollback event. A rolled-back candidate cannot be resumed or
reactivated until a newer configuration has been staged and explicitly activated.
All registration, activation, pause/resume, stable selection, rollback,
deactivation, and deletion changes are durably audited and logged without
request content.

## Chat continuity and streaming

Experimental sessions rebuild the selected model from a copy of the existing
conversation history. Only successful attempts replace the active chat session;
failed candidates cannot append partial or duplicate turns to the user's
conversation. The stable retry uses the snapshot captured during assignment,
even if an administrator deletes the experiment while the request is in flight.
Deactivation migrates the next turn back to the default model with its history
intact. Experimental requests neither read nor write the shared response caches.

Non-streaming responses expose `X-Experiment-Id`, `X-Experiment-Version`,
`X-Experiment-Variant`, and `X-Experiment-Rolled-Back`. The version identifies the
configuration that actually generated the reply; controls may therefore report
an older stable artifact revision than the active canary configuration.

Activated experiments use a buffered SSE path: metadata is emitted immediately,
then a complete, safety-processed answer is replayed as content deltas followed
by the usual `done` event with experiment metadata. Buffering delays the first
content delta for experimental traffic, but prevents a failed treatment from
leaking partial output or producing two overlapping answers during fallback.
Default, non-experimental streams retain their original live streaming path.
Disconnect cancellation does not commit partial experimental history or count
as a model failure.

## State storage

`EXPERIMENT_DB_PATH` defaults to `data/experiments.sqlite3`. Configure a writable
persistent local volume to retain snapshots and audit history across container
replacement. SQLite transactions coordinate workers that share that database
file, and all workers must use the same path. Separate replicas with independent
files do not share rollback decisions; coordinate those deployments before
enabling an experiment across them. Runtime database files are excluded from
Git and Docker build contexts.

A runtime database read failure selects the default generation path, preserving
history. A metrics-write failure is logged without failing an otherwise valid
answer. Stable/control model failures use the existing service error handling.

## Offline validation

```bash
pytest -q tests/test_experiments.py tests/test_experiment_api.py
pytest -q tests/test_prompt_registry.py tests/test_chat_api.py tests/test_stream_cache.py
ruff check .
ruff format --check .
mypy .
```

Tests cover rollout percentages, legacy A/B compatibility, immutable revisions,
model/prompt/settings selection, stable fallback, cache isolation, error and
latency triggers, stale observations, cross-worker coordination, restart
recovery, authenticated controls, streaming, cancellation, and conversation
history preservation.
