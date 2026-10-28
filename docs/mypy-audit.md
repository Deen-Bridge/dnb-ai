# Mypy override audit

The audit applies to the `dev` configuration. Application modules must remain
visible to `mypy .`; there are no `ignore_errors` overrides. The historical
`store` exemption mentioned in issue #413 was already removed on this branch.

| Previous exemption | Decision | Reason |
| --- | --- | --- |
| `google.*` | Narrow to the exact `google` namespace root, `google.generativeai` and its children | The legacy Gemini SDK has no typing marker; other Google libraries must remain checked. |
| `stellar_sdk.*` | Remove | The installed SDK supplies typing metadata. |
| `redis.*` | Remove | Redis 5 supplies typing metadata. |
| `yaml.*` | Remove | Install PyYAML stubs as a development dependency. |
| `firebase_admin`, `firebase_admin.*` | Retain | The installed Firebase Admin SDK has no typing marker or bundled stubs. |
| `locust`, `locust.*` | Retain | Locust is only installed by the performance job; the lint job still checks the load test source. |
| `prometheus_fastapi_instrumentator`, children | Remove | The installed library supplies typing metadata. |
| `prometheus_client`, children | Remove | The installed library supplies typing metadata. |
| `PIL`, `PIL.*` | Remove | The existing development dependencies provide Pillow stubs. |

The `main` override (`disallow_untyped_defs = true`) adds checking rather than
suppressing it and remains enabled. Other modules are checked under the normal
mypy defaults; requiring complete annotations across every legacy module is a
separate migration. Namespace packages, explicit package bases, and `mypy_path`
resolve this repository's unpackaged script directories rather than disable
checking. The only excluded paths are local virtual environments.

`warn_unused_ignores` is enabled so stale inline suppressions are reported during
CI. Remaining inline ignores identify a specific error code and a concrete
external limitation or deliberately invalid test input.

To repeat the audit, install both requirements files, run `mypy .`, and review
`pyproject.toml` plus inline `type: ignore` directives. Do not add application
modules to the missing-import exceptions to mask real errors.
