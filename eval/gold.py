"""Read and write the gold set, and fetch the indexed chunks."""
import csv
from pathlib import Path

from retrieval.retrieve import get_collection

GOLD_CSV = Path("eval/gold.csv")
RESULTS_DIR = Path("eval/results")


def load_gold() -> list[dict]:
    """Every gold row as a dict keyed by column name."""
    with GOLD_CSV.open(newline="") as f:
        return list(csv.DictReader(f))


def save_rows(path: Path, rows: list[dict]) -> None:
    """Write rows to a CSV, using the first row's keys as the header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def answerable(rows: list[dict]) -> list[dict]:
    """Only the rows that have an answer in a label."""
    return [r for r in rows if r["answerable"] == "y"]


def chunks_for(drug: str) -> list[dict]:
    """All indexed chunks for one drug: id, text and section."""
    got = get_collection().get(where={"drug": drug}, include=["documents", "metadatas"])
    return [
        {"id": i, "text": t, "section": m["section"]}
        for i, t, m in zip(got["ids"], got["documents"], got["metadatas"])
    ]
