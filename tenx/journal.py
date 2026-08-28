"""Append-only JSONL reasoning trail. Every layer logs full reasoning here,
not just outcomes (spec: docs/BUILD_PLAN.md section 4.6)."""
import json
from datetime import datetime, timezone
from pathlib import Path


class Journal:
    def __init__(self, path: Path | str):
        self.path = Path(path)

    def log(self, run_id: str, stage: str, payload: dict) -> dict:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "stage": stage,
            "payload": payload,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")
        return record

    def read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        with self.path.open() as f:
            return [json.loads(line) for line in f if line.strip()]
