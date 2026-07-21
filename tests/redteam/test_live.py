import json
import os
from pathlib import Path

import pytest
import yaml


pytestmark = pytest.mark.skipif(
    os.getenv("SAFETY_LIVE_TESTS") != "1",
    reason="live red-team audit is opt-in via SAFETY_LIVE_TESTS=1",
)


def test_live_classifier_audit():
    import google.generativeai as genai

    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel("gemini-2.5-flash-preview-05-20")
    cases = yaml.safe_load(
        Path(__file__).with_name("prompts.yaml").read_text(encoding="utf-8")
    )
    for case in cases:
        response = model.generate_content(
            "Return strict JSON with category_id and action for this expected safety audit. "
            f"Known policy ids: DB-SAFE-001..004 or null. Text: {case['prompt']}",
            generation_config={"temperature": 0, "response_mime_type": "application/json"},
        )
        actual = json.loads(response.text)
        assert actual["category_id"] == case["category_id"]
        assert actual["action"] == case["action"]
