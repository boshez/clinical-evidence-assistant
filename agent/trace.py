"""Reasoning trace: one small record per node, saved for the UI to show."""
import json
from datetime import datetime
from pathlib import Path

TRACE_DIR = Path("data/traces")


def entry(node: str, summary: str, **detail) -> list[dict]:
    """Build one trace record, wrapped in a list.

    WHY a list: the state's trace field is append-only, and LangGraph appends
    by adding lists together.
    """
    return [{
        "node": node,
        "summary": summary,
        "detail": detail,
        "at": datetime.now().isoformat(timespec="milliseconds"),
    }]


def save(run_id: str, trace: list[dict]) -> Path:
    """Write the trace as JSON Lines (one record per line) and return the path."""
    # WHY JSON Lines: the UI can read it line by line, and each line is a
    # complete record even if a run stops half-way.
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    path = TRACE_DIR / f"{run_id}.jsonl"
    with path.open("w") as f:
        for step, record in enumerate(trace, start=1):
            f.write(json.dumps({"step": step, **record}) + "\n")
    return path
