"""Browser-level RTL and mixed-script rendering checks (#878).

These tests serve the committed reference client from ``tests/browser/`` on
loopback and drive it with Playwright. They are skipped when Playwright or a
Chromium build is unavailable, so ``pytest -q tests/`` stays green on a
contributor machine without browsers. CI installs Chromium and runs them.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterator
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

sync_api = pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
from PIL import Image, ImageChops  # noqa: E402

BROWSER_DIR = Path(__file__).resolve().parent
FIXTURES_DIR = BROWSER_DIR / "fixtures"
ARTIFACTS_DIR = BROWSER_DIR / "artifacts"
BASELINES_DIR = BROWSER_DIR / "baselines"
TRACES_DIR = ARTIFACTS_DIR / "traces"

VIEWPORTS = {
    "desktop": {"width": 1280, "height": 800},
    "mobile": {"width": 390, "height": 844},
}

MAX_DIFF_RATIO = 0.05


CASES = [
    {
        "name": "arabic",
        "fixture": "arabic.json",
        "direction": "rtl",
        "first": "ما",
        "last": "الفرج",
        "punctuation": "؟",
        "latin": "sabr",
    },
    {
        "name": "urdu",
        "fixture": "urdu.json",
        "direction": "rtl",
        "first": "صبر",
        "last": "ساتھ",
        "punctuation": "؟",
        "latin": "sabr",
    },
    {
        "name": "swahili_english",
        "fixture": "swahili_english.json",
        "direction": "ltr",
        "first": "Subira",
        "last": "virtue",
        "punctuation": ".",
        "latin": "patience",
    },
    {
        "name": "mixed",
        "fixture": "mixed.json",
        "direction": "rtl",
        "first": "الصبر",
        "last": "key",
        "punctuation": "؟",
        "latin": "sabr",
    },
]


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - stdlib signature
        return


@pytest.fixture(scope="session")
def base_url() -> Iterator[str]:
    handler = lambda *args, **kwargs: _QuietHandler(*args, directory=str(BROWSER_DIR), **kwargs)  # noqa: E731
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


@pytest.fixture(scope="session")
def browser() -> Iterator[Any]:
    try:
        playwright = sync_api.sync_playwright().start()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Playwright could not start: {exc}")
    try:
        chromium = playwright.chromium.launch()
    except Exception as exc:  # pragma: no cover - environment dependent
        playwright.stop()
        pytest.skip(f"Chromium is not installed for Playwright: {exc}")
    try:
        yield chromium
    finally:
        chromium.close()
        playwright.stop()


def _fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES_DIR / name).read_text(encoding="utf-8"))


def _rect_for(page: Any, text: str) -> dict[str, float]:
    return page.evaluate(
        """(needle) => {
            const node = document.querySelector('#bidi-probe').firstChild;
            const start = node.textContent.indexOf(needle);
            if (start < 0) { throw new Error('text not found: ' + needle); }
            const range = document.createRange();
            range.setStart(node, start);
            range.setEnd(node, start + needle.length);
            const rect = range.getBoundingClientRect();
            return {x: rect.x, y: rect.y, width: rect.width, height: rect.height};
        }""",
        text,
    )


def _focused(page: Any) -> str:
    return page.evaluate("() => document.activeElement && document.activeElement.id")


def _safe_id(value: str) -> str:
    return "".join(char if char.isalnum() or char in "_-" else "-" for char in value)


def _compare_screenshot(actual: Path, baseline: Path, update: bool) -> None:
    if update or not baseline.exists():
        if not update:
            pytest.fail(f"Missing visual baseline {baseline}. Re-run with --update-baselines after review.")
        baseline.parent.mkdir(parents=True, exist_ok=True)
        actual.replace(baseline)
        return

    with Image.open(actual) as actual_image, Image.open(baseline) as baseline_image:
        if actual_image.size != baseline_image.size:
            pytest.fail(f"Screenshot size changed: {actual_image.size} != {baseline_image.size}")
        diff = ImageChops.difference(actual_image.convert("RGB"), baseline_image.convert("RGB"))
        histogram = diff.convert("L").point(lambda value: 255 if value > 16 else 0).histogram()
        changed = histogram[255]
        total = actual_image.size[0] * actual_image.size[1]
        ratio = changed / total
    assert ratio <= MAX_DIFF_RATIO, (
        f"Visual regression: {ratio:.2%} of pixels changed (limit {MAX_DIFF_RATIO:.0%}). "
        f"Inspect {actual} against {baseline}; update baselines only with reviewer approval."
    )


@pytest.mark.parametrize("viewport_name", sorted(VIEWPORTS))
@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_rtl_rendering(
    browser: Any,
    base_url: str,
    case: dict[str, str],
    viewport_name: str,
    request: pytest.FixtureRequest,
) -> None:
    payload = _fixture(case["fixture"])
    update = bool(request.config.getoption("--update-baselines")) or os.getenv("UPDATE_RTL_BASELINES") == "1"
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    TRACES_DIR.mkdir(parents=True, exist_ok=True)

    context = browser.new_context(viewport=VIEWPORTS[viewport_name], device_scale_factor=1)
    context.tracing.start(screenshots=True, snapshots=True, sources=True)
    page = context.new_page()
    screenshot = ARTIFACTS_DIR / f"{case['name']}_{viewport_name}.png"
    trace = TRACES_DIR / f"{case['name']}_{viewport_name}.zip"
    try:
        page.goto(f"{base_url}/index.html?fixture={case['fixture']}")
        page.wait_for_function("() => window.__ready === true")
        assert page.evaluate("() => window.__error || null") is None

        answer = page.locator("#answer")
        assert answer.get_attribute("dir") == case["direction"]
        assert answer.evaluate("el => getComputedStyle(el).direction") == case["direction"]
        assert answer.text_content() == payload["response"]
        assert case["punctuation"] in (answer.text_content() or "")

        first = _rect_for(page, case["first"])
        last = _rect_for(page, case["last"])
        punctuation = _rect_for(page, case["punctuation"])
        latin = _rect_for(page, case["latin"])
        if case["direction"] == "rtl":
            assert first["x"] > last["x"], "first logical run must render to the right of the last"
            assert first["x"] > latin["x"] > last["x"], "embedded Latin run must keep its bidi position"
            assert punctuation["x"] <= last["x"] + last["width"], "Arabic punctuation must stay at the visual end"
        else:
            assert first["x"] < last["x"], "first logical run must render to the left of the last"
            assert first["x"] < latin["x"] < last["x"]
            assert punctuation["x"] >= last["x"], "Latin punctuation must stay at the visual end"

        links = page.locator("a.citation-link")
        assert links.count() == len(payload["citations"])
        for index, citation in enumerate(payload["citations"]):
            link = links.nth(index)
            assert link.get_attribute("data-reference") == citation["reference"]
            assert link.text_content() == citation["reference"]
            figure = page.locator(f"#citation-block-{_safe_id(citation['reference'])}")
            quote = figure.locator("blockquote")
            assert quote.text_content() == citation["quote"]
            assert quote.get_attribute("dir") == citation.get("quote_direction", "rtl")
            quote_box = quote.bounding_box()
            link_box = link.bounding_box()
            assert quote_box is not None and link_box is not None
            assert abs(link_box["y"] - quote_box["y"]) <= quote_box["height"] + 80

        page.screenshot(path=str(screenshot), full_page=True)
        _compare_screenshot(
            screenshot,
            BASELINES_DIR / f"{case['name']}_{viewport_name}.png",
            update=update,
        )

        aria = page.locator("body").aria_snapshot()
        positions = [aria.find(citation["reference"]) for citation in payload["citations"]]
        assert all(position >= 0 for position in positions)
        assert positions == sorted(positions), "citation reading order must follow the answer"

        page.locator("#copy-button").focus()
        assert _focused(page) == "copy-button"
        for citation in payload["citations"]:
            page.keyboard.press("Tab")
            assert _focused(page) == f"citation-{_safe_id(citation['reference'])}"

        page.locator("#copy-button").click()
        copied = page.locator("#copy-payload").text_content()
        assert copied == payload["response"]
        assert page.evaluate("() => window.__copiedText") == payload["response"]
    except Exception:
        context.tracing.stop(path=str(trace))
        raise
    else:
        context.tracing.stop()
    finally:
        context.close()
