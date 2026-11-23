# PolinRider Repository & Supply-Chain Security Audit (#521)

## Scope Reviewed
- **Branches & Refs**: `main`, `dev`
- **Manifests & Lockfiles**: `requirements.txt`, `requirements-dev.txt`, `Dockerfile`, `.github/workflows/*.yml`
- **Editor & Task Configs**: `.vscode/tasks.json` (verified absent), `pyproject.toml`
- **Static & Binary Assets**: Checked for unexpected `.woff`, `.ttf`, `.otf`, `.wasm`, or executable binaries with appended payloads.

## Findings
- **Zero PolinRider indicators detected**: No `runOn: "folderOpen"` tasks, no obfuscated `eval(atob(...))` tails in config files, and no suspicious binary font payloads.
- Runtime and development dependencies are installed from committed hash-verified lockfiles. CI audits both sets and retains JSON scan results, CycloneDX SBOMs, and deterministic build provenance for 90 days; release provenance is attested to the source commit.

## Audit Limits & Remediation Policy
- Static non-executing inspection only; third-party PyPI wheels are pinned and hash-verified by `requirements.lock` and `requirements-dev.lock` in CI and deployment.
