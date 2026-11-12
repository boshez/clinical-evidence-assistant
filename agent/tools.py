"""The agent's three tools. Each wraps code built in an earlier step."""
import csv
from datetime import date
from functools import lru_cache
from pathlib import Path

from neo4j import Driver

from graph.connect import get_driver
from graph.safety import patient_drug_safety
from retrieval.retrieve import retrieve

RISK_CSV = Path("data/patient_risk_scores.csv")


@lru_cache(maxsize=1)
def driver() -> Driver:
    """Open the Neo4j connection once and reuse it."""
    return get_driver()


def guideline_lookup(query: str, k: int = 5) -> list[dict]:
    """Tool 1: the k label chunks closest to the query, best first."""
    return retrieve(query, k)


def safety_check(patient_id: str, drug: str, as_of: str) -> dict:
    """Tool 2: contraindications and interactions active on as_of.

    Raises ValueError for an unknown patient or drug, or a bad date.
    """
    return patient_drug_safety(
        driver(), patient_id, drug, date.fromisoformat(as_of)
    )


def readmission_risk(patient_id: str) -> dict:
    """Tool 3: the patient's score from their most recent admission."""
    with RISK_CSV.open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["patient_id"] == patient_id]
    if not rows:
        # WHY this wording: the hand-made TEST_ patients are in the graph but
        # not in this file, so "unknown patient" would be misleading.
        raise ValueError(f"no readmission score on file for patient {patient_id}")
    # WHY the latest admission: the file has one row per hospital stay, and
    # "this patient's risk" most naturally means the most recent one.
    # ISO dates sort correctly as plain text.
    latest = max(rows, key=lambda r: r["admission_date"])
    return {
        "patient_id": patient_id,
        "score": round(float(latest["readmission_risk_score"]), 3),
        "admission_date": latest["admission_date"],
        "admissions_on_file": len(rows),
    }
