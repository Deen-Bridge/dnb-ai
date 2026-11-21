import json

import incident_replay as ir


def test_fixture_is_sanitized_and_well_formed():
    incidents = ir.load_incidents()
    ids = [i["id"] for i in incidents]
    assert len(ids) == len(set(ids))
    for inc in incidents:
        assert {"id", "description", "response", "expect_detected"} <= inc.keys()
        text = json.dumps(inc).lower()
        assert "api_key" not in text and "sk-" not in text and "@" not in text


def test_baseline_has_owner_and_rollback():
    base = ir.load_baseline()
    assert base["owner"] and base["rollback"]
    assert 0 < base["min_pass_rate"] <= 1


def test_replay_meets_threshold():
    ok, results, messages = ir.check()
    assert ok, "\n".join(messages)
    assert results["total"] >= ir.load_baseline()["min_incidents"]


def test_regression_is_reported_actionably(tmp_path):
    fixture = tmp_path / "bad.jsonl"
    fixture.write_text(
        json.dumps(
            {
                "id": "INC-X",
                "description": "clean text wrongly expected to be flagged",
                "response": "Prophet Muhammad passed away in 632 CE.",
                "expect_detected": True,
                "expect_type": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    ok, _, messages = ir.check(fixture=fixture)
    assert not ok
    assert any("INC-X" in m for m in messages)


def test_main_exit_code_zero():
    assert ir.main() == 0
