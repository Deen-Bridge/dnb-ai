"""Tests for hardened container deployment contract and Render configuration (#880).

Validates:
- render.yaml specifies hardened Docker runtime (not Python buildpack)
- render.yaml configures Dockerfile path, context, and readiness probe
- Dockerfile adheres to security hardening: pinned slim base, non-root user,
  purged compilers, non-writable root with scoped write paths, and healthcheck
- Liveness (/ping) and readiness (/health) probes are properly declared
- CI workflow builds the same Dockerfile, records commit and digest, scans vulnerabilities,
  and verifies non-root runtime execution and write paths
"""

from __future__ import annotations

import ast
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
RENDER_YAML_PATH = REPO_ROOT / "render.yaml"
DOCKERFILE_PATH = REPO_ROOT / "Dockerfile"
CI_WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "ci.yml"
MAIN_PY_PATH = REPO_ROOT / "main.py"


def test_render_yaml_uses_hardened_docker_runtime() -> None:
    """Verify render.yaml switches from Python buildpack to hardened Docker runtime."""
    assert RENDER_YAML_PATH.exists(), "render.yaml must exist"
    with open(RENDER_YAML_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    assert "services" in config, "render.yaml must define services"
    services = config["services"]
    assert len(services) >= 1, "At least one service must be defined"

    web_svc = services[0]
    assert web_svc.get("type") == "web"
    assert web_svc.get("name") == "deenbridge-ai"

    # Runtime must be docker (not python buildpack)
    assert web_svc.get("runtime") == "docker", "render.yaml must use `runtime: docker`"
    assert web_svc.get("env") != "python", "`env: python` must be replaced with `runtime: docker`"

    # Dockerfile path and context must be explicitly set
    assert web_svc.get("dockerfilePath") == "./Dockerfile"
    assert web_svc.get("dockerContext") == "."

    # buildCommand and startCommand must not be present when using Dockerfile
    assert "buildCommand" not in web_svc, "buildCommand should not be set for Docker runtime"
    assert "startCommand" not in web_svc, "startCommand should not be set for Docker runtime"

    # Readiness probe path must match service contract
    assert web_svc.get("healthCheckPath") == "/health"


def test_dockerfile_hardened_container_contract() -> None:
    """Verify Dockerfile adheres to all required security hardening invariants."""
    assert DOCKERFILE_PATH.exists(), "Dockerfile must exist"
    content = DOCKERFILE_PATH.read_text(encoding="utf-8")

    # 1. Pinned base image (slim)
    assert "FROM python:3.12-slim" in content, "Must pin specific slim Python base image"

    # 2. Attack surface reduction: compiler removal and apt cleanup
    assert "apt-get purge -y --auto-remove gcc" in content, "Must purge gcc compiler after build"
    assert "rm -rf /var/lib/apt/lists/*" in content, "Must clean up apt caches to minimize image size"

    # 3. Non-root user creation and switch
    assert "addgroup --system app" in content, "Must create dedicated system group 'app'"
    assert "adduser --system --ingroup app app" in content, "Must create system user 'app'"
    assert "USER app" in content, "Must switch to non-root user 'app'"

    # 4. Scoped writable storage for non-root user
    assert "mkdir -p /app/data/review" in content, "Must create review data directory"
    assert "chown -R app:app /app/data" in content, "Must own /app/data to user app"

    # 5. Container healthcheck instruction for liveness
    assert "HEALTHCHECK" in content, "Must define container HEALTHCHECK instruction"
    assert "/ping" in content, "HEALTHCHECK must check liveness probe /ping"

    # 6. Port and entrypoint
    assert "EXPOSE 8000" in content, "Must expose application port 8000"
    assert "uvicorn main:app" in content, "Entrypoint CMD must invoke uvicorn"


def test_main_py_declares_liveness_and_readiness_probes() -> None:
    """Verify main.py declares /ping (liveness) and /health (readiness)."""
    assert MAIN_PY_PATH.exists(), "main.py must exist"
    tree = ast.parse(MAIN_PY_PATH.read_text(encoding="utf-8"))

    route_paths: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for decorator in node.decorator_list:
                if (
                    isinstance(decorator, ast.Call)
                    and isinstance(decorator.func, ast.Attribute)
                    and decorator.func.attr in ("get", "post")
                ):
                    if (
                        decorator.args
                        and isinstance(decorator.args[0], ast.Constant)
                        and isinstance(decorator.args[0].value, str)
                    ):
                        route_paths.add(decorator.args[0].value)

    assert "/ping" in route_paths, "/ping liveness probe must be declared in main.py"
    assert "/health" in route_paths, "/health readiness probe must be declared in main.py"


def test_ci_workflow_builds_and_scans_same_docker_image() -> None:
    """Verify CI workflow builds the Dockerfile, records digest, and scans for vulnerabilities."""
    assert CI_WORKFLOW_PATH.exists(), ".github/workflows/ci.yml must exist"
    with open(CI_WORKFLOW_PATH, encoding="utf-8") as f:
        ci_config = yaml.safe_load(f)

    jobs = ci_config.get("jobs", {})
    assert "build-and-scan-image" in jobs, "CI must contain `build-and-scan-image` job (#880)"

    job = jobs["build-and-scan-image"]
    assert "lint-and-test" in job.get("needs", []), "`build-and-scan-image` should run with/after `lint-and-test`"

    steps = job.get("steps", [])
    step_names = [s.get("name", "") for s in steps]
    step_runs = [s.get("run", "") for s in steps if "run" in s]
    combined_runs = "\n".join(step_runs)

    # 1. Docker Buildx setup
    assert any("setup-buildx-action" in s.get("uses", "") for s in steps), "Must set up Docker Buildx"

    # 2. Build production container
    build_step = next((s for s in steps if "build-push-action" in s.get("uses", "")), None)
    assert build_step is not None, "Must build container image via build-push-action"
    assert build_step.get("with", {}).get("file") == "./Dockerfile"

    # 3. Commit and image digest recording
    assert any("Record commit and container image digest" in name for name in step_names)
    assert "IMAGE_DIGEST" in combined_runs
    assert "image-digest.txt" in combined_runs

    # 4. Vulnerability scanning (Trivy)
    assert any("trivy-action" in s.get("uses", "") for s in steps), "Must scan image using Trivy"

    # 5. Staging verification: non-root check & write paths
    assert any("Verify non-root execution" in name for name in step_names)
    assert "whoami" in combined_runs
    assert "/app/data/review" in combined_runs
    assert "/ping" in combined_runs
    assert "/health" in combined_runs
