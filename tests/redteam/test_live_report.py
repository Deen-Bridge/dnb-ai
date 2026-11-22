from pathlib import Path

from tests.redteam.live_report import MAX_LIVE_CASES, build_report, redact

ROOT = Path(__file__).resolve().parents[2]


def _r(passed, actual="x"):
    return {"case": "c", "expected": "e", "actual": actual, "passed": passed}


def test_threshold_blocks_on_failure():
    report = build_report([_r(True), _r(False)], provider="p", model="m", policy_version="1")
    assert report["decision"] == "block"
    assert report["policy_version"] == "1" and report["model"] == "m"


def test_all_pass_allows_release():
    assert build_report([_r(True)], provider="p", model="m", policy_version="1")["decision"] == "pass"


def test_report_redacts_secrets():
    report = build_report(
        [_r(False, "key AIza" + "A" * 30)], provider="p", model="m", policy_version="1"
    )
    assert "AIza" not in str(report)
    assert "[REDACTED]" in redact("Bearer abc123")


def test_budget_cap_and_spec_present():
    assert MAX_LIVE_CASES > 0
    assert (ROOT / "safety" / "LIVE_EVAL.md").exists()
