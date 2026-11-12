"""Load synthetic patients, with dated conditions and medications, into Neo4j.

Patient IDs and ages come from Project 1's CSV. That CSV has no condition or
drug names, so those are generated here with a fixed random seed. Everything in
this file is synthetic; none of it describes a real person.
"""
import csv
import random
from datetime import date, timedelta
from pathlib import Path

from neo4j import Driver

from graph.connect import get_driver

PATIENTS_CSV = Path("data/patient_risk_scores.csv")
APPROVED_CSV = Path("data/edges_approved.csv")
SAMPLE_SIZE = 200
SEED = 42  # WHY: a fixed seed gives the same 200 patients on every run.

# Long-term conditions stay open (no resolved date).
CHRONIC = [
    "hypertension", "heart failure", "type 2 diabetes", "hyperlipidemia",
    "coronary artery disease", "angina", "osteoarthritis",
    "gastroesophageal reflux disease",
]
# Short-term conditions usually get a resolved date. These are the ones that
# cause false alarms if a safety check ignores dates.
TRANSIENT = [
    "hyperkalemia", "active bleeding", "edema", "metabolic acidosis",
    "decompensated heart failure", "severe bradycardia", "anuria",
]
EARLIEST = date(2023, 1, 1)
LATEST = date(2026, 6, 30)

# Hand-made patients for the temporal tests. Dates are chosen around the test
# date 2026-06-01 so that each one exercises exactly one rule.
TEST_PATIENTS = [
    {"patient_id": "TEST_ACTIVE", "age": 60,
     "conditions": [("hyperkalemia", "2026-05-01", None)], "drugs": []},
    {"patient_id": "TEST_RESOLVED", "age": 60,
     "conditions": [("hyperkalemia", "2025-01-10", "2025-02-01")], "drugs": []},
    {"patient_id": "TEST_FUTURE", "age": 60,
     "conditions": [("hyperkalemia", "2026-07-15", None)], "drugs": []},
    {"patient_id": "TEST_BOUNDARY", "age": 60,
     "conditions": [("hyperkalemia", "2026-05-01", "2026-06-01")], "drugs": []},
    {"patient_id": "TEST_TAKES_NOW", "age": 60,
     "conditions": [], "drugs": [("warfarin", "2026-01-01", None)]},
    {"patient_id": "TEST_TAKES_STOPPED", "age": 60,
     "conditions": [], "drugs": [("warfarin", "2025-01-01", "2025-03-01")]},
]


def sample_patients(rng: random.Random) -> list[dict]:
    """Pick SAMPLE_SIZE patients from the CSV, with each one's age."""
    ages: dict[str, int] = {}
    with PATIENTS_CSV.open() as f:
        for row in csv.DictReader(f):  # one row per admission; keep one per patient
            ages[row["patient_id"]] = int(float(row["age"]))
    chosen = rng.sample(sorted(ages), SAMPLE_SIZE)
    return [{"patient_id": pid, "age": ages[pid]} for pid in chosen]


def treating_drugs() -> dict[str, list[str]]:
    """Map each condition to the drugs with an approved TREATS edge to it."""
    # WHY: patients should take drugs that treat what they have, so the
    # synthetic data looks like real prescribing instead of pure noise.
    treats: dict[str, list[str]] = {}
    with APPROVED_CSV.open() as f:
        for row in csv.DictReader(f):
            if row["approved"] == "y" and row["relation"] == "TREATS":
                treats.setdefault(row["target"], []).append(row["drug"])
    return treats


def random_day(rng: random.Random, start: date, end: date) -> date:
    """A random date between start and end, inclusive."""
    return start + timedelta(days=rng.randint(0, (end - start).days))


def make_conditions(rng: random.Random) -> list[tuple]:
    """1-3 chronic conditions (open) and 0-2 transient ones (mostly resolved)."""
    conditions = []
    for name in rng.sample(CHRONIC, rng.randint(1, 3)):
        conditions.append((name, random_day(rng, EARLIEST, LATEST).isoformat(), None))
    for name in rng.sample(TRANSIENT, rng.randint(0, 2)):
        onset = random_day(rng, EARLIEST, LATEST)
        resolved = None
        if rng.random() < 0.8:  # most short-term conditions have ended
            resolved = (onset + timedelta(days=rng.randint(7, 120))).isoformat()
        conditions.append((name, onset.isoformat(), resolved))
    return conditions


def make_drugs(rng: random.Random, conditions: list[tuple],
               treats: dict[str, list[str]]) -> list[tuple]:
    """For most treatable conditions, start one treating drug after onset."""
    drugs: dict[str, tuple] = {}
    for name, onset, _ in conditions:
        if name not in treats or rng.random() > 0.8:
            continue
        drug = rng.choice(treats[name])
        start = date.fromisoformat(onset) + timedelta(days=rng.randint(0, 30))
        stop = None
        if rng.random() < 0.25:  # a quarter of prescriptions were later stopped
            stop = (start + timedelta(days=rng.randint(30, 365))).isoformat()
        drugs.setdefault(drug, (drug, start.isoformat(), stop))
    return list(drugs.values())


def build_patients() -> list[dict]:
    """The 200 sampled patients plus the hand-made test patients."""
    rng = random.Random(SEED)
    treats = treating_drugs()
    patients = sample_patients(rng)
    for patient in patients:
        patient["conditions"] = make_conditions(rng)
        patient["drugs"] = make_drugs(rng, patient["conditions"], treats)
    return patients + TEST_PATIENTS


def to_rows(patients: list[dict], key: str, names: tuple[str, str, str]) -> list[dict]:
    """Flatten each patient's conditions or drugs into one row per edge."""
    target, first, last = names
    return [
        {"patient_id": p["patient_id"], target: item[0], first: item[1], last: item[2]}
        for p in patients for item in p[key]
    ]


def load_patients(driver: Driver, patients: list[dict]) -> None:
    """Replace all Patient nodes and their dated edges."""
    # WHY: delete first, so a re-run never leaves edges from an older version.
    driver.execute_query("MATCH (p:Patient) DETACH DELETE p")
    driver.execute_query(
        """
        UNWIND $rows AS row
        MERGE (p:Patient {patient_id: row.patient_id})
        SET p.age = row.age, p.synthetic = true
        """,
        rows=[{"patient_id": p["patient_id"], "age": p["age"]} for p in patients],
    )
    # WHY: date() turns "2026-05-01" into a real date, so Cypher can compare
    # dates properly. date(null) stays null, which means "still ongoing".
    driver.execute_query(
        """
        UNWIND $rows AS row
        MATCH (p:Patient {patient_id: row.patient_id}), (c:Condition {name: row.condition})
        MERGE (p)-[h:HAS_CONDITION]->(c)
        SET h.onset = date(row.onset), h.resolved = date(row.resolved)
        """,
        rows=to_rows(patients, "conditions", ("condition", "onset", "resolved")),
    )
    driver.execute_query(
        """
        UNWIND $rows AS row
        MATCH (p:Patient {patient_id: row.patient_id}), (d:Drug {name: row.drug})
        MERGE (p)-[t:TAKES]->(d)
        SET t.start = date(row.start), t.stop = date(row.stop)
        """,
        rows=to_rows(patients, "drugs", ("drug", "start", "stop")),
    )


if __name__ == "__main__":
    all_patients = build_patients()
    with get_driver() as driver:
        load_patients(driver, all_patients)
        records, _, _ = driver.execute_query(
            """
            MATCH (p:Patient)
            OPTIONAL MATCH (p)-[h:HAS_CONDITION]->()
            WITH count(DISTINCT p) AS patients, count(h) AS conditions
            MATCH (:Patient)-[t:TAKES]->()
            RETURN patients, conditions, count(t) AS medications
            """
        )
        row = records[0]
        print(f"patients: {row['patients']}")
        print(f"HAS_CONDITION edges: {row['conditions']}")
        print(f"TAKES edges: {row['medications']}")
