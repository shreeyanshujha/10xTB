import json

from tenx.journal import Journal


def test_log_appends_jsonl_records(tmp_path):
    j = Journal(tmp_path / "journal.jsonl")
    rec1 = j.log("run-1", "data_pull", {"ticker": "NVDA", "rows": 40})
    rec2 = j.log("run-1", "signal", {"action": "BUY"})

    lines = (tmp_path / "journal.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]
    assert parsed[0]["stage"] == "data_pull"
    assert parsed[0]["run_id"] == "run-1"
    assert parsed[0]["payload"]["ticker"] == "NVDA"
    assert parsed[1]["payload"]["action"] == "BUY"
    assert rec1["ts"] <= rec2["ts"]
    assert rec1["ts"].endswith("+00:00")


def test_read_all_round_trips(tmp_path):
    j = Journal(tmp_path / "sub" / "journal.jsonl")  # parent dir auto-created
    j.log("run-1", "a", {})
    j.log("run-2", "b", {"x": 1})
    records = j.read_all()
    assert [r["run_id"] for r in records] == ["run-1", "run-2"]


def test_read_all_missing_file_is_empty(tmp_path):
    assert Journal(tmp_path / "nope.jsonl").read_all() == []
