import json
import os
from pathlib import Path

import pytest
import yaml

from tests.redteam.live_report import MAX_LIVE_CASES, build_report, dumps

ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(
    os.getenv("SAFETY_LIVE_TESTS") != "1",
    reason="live red-team audit is opt-in via SAFETY_LIVE_TESTS=1",
)


def test_live_classifier_audit():
    import google.generativeai as genai

    model_name = os.getenv("SAFETY_LIVE_MODEL", "gemini-2.5-flash-preview-05-20")
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel(model_name)
    cases = yaml.safe_load(Path(__file__).with_name("prompts.yaml").read_text(encoding="utf-8"))
    cases = cases[:MAX_LIVE_CASES]  # budget cap; fixtures are synthetic only
    policy = yaml.safe_load((ROOT / "safety" / "policy.yaml").read_text(encoding="utf-8"))
    results = []
    for case in cases:
        response = model.generate_content(
            "Return strict JSON with category_id and action for this expected safety audit. "
            f"Known policy ids: DB-SAFE-001..004 or null. Text: {case['prompt']}",
            generation_config={"temperature": 0, "response_mime_type": "application/json"},
        )
        actual = json.loads(response.text)
        results.append(
            {
                "case": case.get("id", case["prompt"][:40]),
                "expected": f"{case['category_id']}/{case['action']}",
                "actual": f"{actual.get('category_id')}/{actual.get('action')}",
                "passed": actual.get("category_id") == case["category_id"]
                and actual.get("action") == case["action"],
            }
        )
    report = build_report(
        results, provider="google-gemini", model=model_name, policy_version=str(policy["version"])
    )
    out = Path(os.getenv("SAFETY_LIVE_REPORT", "safety-live-report.json"))
    out.write_text(dumps(report), encoding="utf-8")
    assert report["decision"] == "pass", f"live safety gate failed; see {out}"
