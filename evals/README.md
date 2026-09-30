# Islamic Answer Evaluations

This directory measures answer quality through the real `POST /chat` path. The
dataset is intentionally small and reviewable: it uses mainstream factual
questions, source-reference questions, high-stakes deflection cases, and
adversarial prompts.

## Run

Start the service with authentication disabled for a local run:

```bash
AUTH_DISABLED=true GEMINI_API_KEY=your_key uvicorn main:app --port 8000
```

Then run the deterministic harness:

```bash
python evals/run.py --url http://localhost:8000 --output evals/reports/local.json
```

The command prints every case, a pass rate for each category, and writes the
same result to a JSON artifact. Add `--samples 3` to ask every question three
times and pass a case only when a strict majority of samples passes. This is
useful for borderline checks because chat generation is nondeterministic.

For an authenticated server, pass `--api-key` or set `EVAL_API_KEY`. For an
in-process ASGI call, use `--direct`; that still invokes the application
handler, but it does not avoid Gemini costs.

## Checks

Deterministic checks always run without an evaluator model. Each dataset entry
can require keywords or regular expressions, forbid patterns, require a
respectful refusal, require a qualified-scholar referral, and require Quran
surah and ayah references. Quran expectations can also require a matching
structured citation in the response's `citations` field. A failed HTTP
request is retained as a failed sample in the artifact instead of being
discarded.

The optional judge is separate:

```bash
GEMINI_API_KEY=your_key python evals/run.py --url http://localhost:8000 --judge
```

It makes a second Gemini call per answer and scores faithfulness to sources,
adab, and scope respect on a 1-5 rubric. Judge errors do not change the
deterministic result; they appear under the sample's `judge` field.

## Cost And Baselines

A one-sample run currently contains 36 paid `/chat` requests. Safety
classification or retries can add model calls inside the service. `--samples 3`
raises that to 108 endpoint requests. `--judge` adds up to one additional
Gemini call per answer, so a one-sample judged run is normally 72 model calls
before retries. Exact dollars depend on the deployed model, prompt length,
output length, and provider pricing; the service's `X-LLM-Cost-USD` response
header is recorded when available.

The committed `reports/baseline.json` is the comparison anchor for the
baseline service. Replace its pending placeholder by running the harness once
against the checked-out baseline commit, then compare later runs with:

```bash
jq '.summary, .categories' evals/reports/baseline.json
jq '.summary, .categories' evals/reports/local.json
diff -u \
  <(jq -S '.summary, .categories' evals/reports/baseline.json) \
  <(jq -S '.summary, .categories' evals/reports/local.json)
```

The baseline report is evidence from one run, not a permanent quality target.
Review case-level changes, especially source-reference and refusal cases,
before changing the system prompt, safety settings, generation configuration,
or model.
