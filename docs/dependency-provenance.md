# Dependency and model provenance

## One runtime dependency set

`requirements.txt`, `requirements-dev.txt`, `requirements-perf.txt`,
`requirements-alignment.txt`, and `requirements-installer.txt` are
human-edited input manifests. Do not install them directly in CI or production.
Their generated, exact, SHA-256-verified lockfiles are:

- `requirements-installer.lock` — pinned pip installer used to bootstrap installs.
- `requirements.lock` — runtime dependencies installed by CI, Docker, and Render.
- `requirements-dev.lock` — development/test tools, constrained by the runtime
  lock so CI cannot silently resolve different runtime versions.
- `requirements-perf.lock` — isolated Locust/performance tooling.
- `requirements-alignment.lock` — optional PyTorch/Transformers inference
  dependencies, constrained by the same runtime lock.

Use Python 3.11.14 from `.python-version` to refresh locks:

```bash
make install-dev
make lock
```

Review and commit all changed lockfiles with the source manifest change. Install
with `--require-hashes`; a missing or mismatched artifact hash must fail rather
than fall back to an unhashed download.

## CI security and reproducibility

The `Dependency Lock, Audit, and Provenance` workflow:

1. Installs the hash-locked installer, runtime, and development sets; the
   performance and optional inference sets remain isolated from production.
2. Installs the runtime lock into two clean environments and byte-compares
   generated manifests. The manifest records the commit, Python/pip/platform,
   exact installed distributions, lockfile digest, approved model configuration,
   and prompt/evaluation baseline digests.
3. Runs `pip-audit` against installer, runtime, development, performance, and
   optional inference locks. Vulnerability reports,
   CycloneDX SBOMs, lockfiles, and the provenance manifest are retained as a
   90-day GitHub Actions artifact, including on a failed scan.
4. Attests the provenance manifest for pushes to `main` and published releases.

The comparison is deterministic for the same commit and build environment. The
manifest also records the platform so rebuilds on different runners are
distinguishable instead of being presented as identical.

## Model/provider baseline

`model_provenance.json` is the reviewed inventory of provider SDK and model
identifiers used by the service, including legacy/preview identifiers that are
still present in code. Prompt templates, the safety policy, Quran reference
index, and versioned evaluation datasets are hashed into each build manifest.
Provider-hosted model weights are not downloaded or controlled by this
repository; the manifest records the exact provider model identifier and SDK
version so a release's inference configuration is auditable.
The optional Hugging Face CTC artifact is disabled by default; when enabled,
`phoneme_model.py` requires an explicit model ID and full 40-character commit
revision and disables remote code execution.

## Artifact access and retention

The CI artifact is named `supply-chain-evidence-<commit-sha>` and retained for
90 days. It contains dependency names/versions, lock hashes, scan findings,
SBOMs, model IDs, and prompt/evaluation hashes—not credentials, prompts,
provider responses, or user data. Release provenance attestations are published
to GitHub's attestation service for verification against the source commit.
