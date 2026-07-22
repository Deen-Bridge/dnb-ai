# Deen Bridge Content-Safety Policy

Policy version: **1.0.0**. The machine-readable source of truth is
[`policy.yaml`](policy.yaml); category wording, examples, actions, guidance,
and failure behavior must be changed there first and reviewed by a maintainer.

## Principles

The assistant provides educational information. It does not impersonate a
scholar, issue binding fatwas, inflame sectarian hostility, or help with harm.
Every non-allow decision records the applicable policy ID and action. User
prompts are never written in full to moderation logs.

## Categories and enforcement

| Policy ID | Category | Normal action | Classifier failure |
|---|---|---|---|
| `DB-SAFE-001` | High-stakes personal rulings | General information with guidance; post-generation scholar disclaimer is mandatory | Fail closed to `allow_with_guidance` |
| `DB-SAFE-002` | Sectarian provocation / takfir | Respectful de-escalating refusal; generator is not called | Fail closed to `refuse` |
| `DB-SAFE-003` | Scholar impersonation | Refuse the role and offer educational help; generator is not called | Fail closed to `refuse` |
| `DB-SAFE-004` | Harmful-intent religious framing | Refuse and provide immediate-danger direction where appropriate; generator is not called | Fail closed to `refuse` |

Benign academic, historical, and comparative questions are explicit near
misses and remain allowed. The deterministic prefilter only identifies
classification candidates; a strict-JSON classifier makes the policy decision.

## Pipeline

1. The prefilter routes possible category matches to classification.
2. Classification returns exactly `category_id`, `confidence`, and `action`.
3. `allow_with_guidance` adds YAML-owned guidance before generation.
4. `refuse` returns the YAML-owned response without calling the generator.
5. The output checker appends the standard scholar referral when required and
   replaces known violating output with the category refusal.
6. Logs record policy ID, action, stages, and added latency—not the full prompt.

## Red-team operation

`pytest -q tests/redteam` is fully offline and is required in CI. Set
`SAFETY_LIVE_TESTS=1` with `GEMINI_API_KEY` to opt into a manual live audit;
live results are intentionally not a merge gate because model behavior can vary.

## Known limitations

Output enforcement is deliberately conservative: regex matches cannot yet
distinguish an endorsed sectarian claim from the same words quoted in order to
refute or study them. A response quoting a prohibited claim may therefore be
replaced with a refusal. This false positive is documented by an offline test;
context-aware output classification is deferred to a follow-up issue.
