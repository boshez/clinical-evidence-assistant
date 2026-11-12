"""Check whether a drug is flagged for a patient on a given date.

The key rule: a condition or a medication only counts if it was ACTIVE on the
as_of date. Active means it had started (onset <= as_of) and had not ended
(resolved is empty, or resolved > as_of). This stops a condition that ended
years ago from raising a false alarm today.
"""
from datetime import date

from neo4j import Driver

# WHY one query for existence: answering "no flags" for a patient or drug that
# is not in the graph would look like reassurance. It must be an error instead.
EXISTS_QUERY = """
OPTIONAL MATCH (p:Patient {patient_id: $patient_id})
OPTIONAL MATCH (d:Drug {name: $drug})
RETURN p IS NOT NULL AS patient_found, d IS NOT NULL AS drug_found
"""

# Conditions the patient had on as_of that the drug is contraindicated in.
CONTRAINDICATION_QUERY = """
MATCH (p:Patient {patient_id: $patient_id})-[h:HAS_CONDITION]->(c:Condition)
      <-[ci:CONTRAINDICATED_IN]-(d:Drug {name: $drug})
WHERE h.onset <= date($as_of)
  AND (h.resolved IS NULL OR h.resolved > date($as_of))
RETURN c.name AS condition, h.onset AS onset, h.resolved AS resolved,
       ci.source AS source
ORDER BY condition
"""

# Drugs the patient was taking on as_of that interact with the drug.
# The INTERACTS_WITH pattern has no arrow, so it matches either direction:
# an interaction written in one drug's label applies to the pair.
INTERACTION_QUERY = """
MATCH (p:Patient {patient_id: $patient_id})-[t:TAKES]->(other:Drug)
      -[i:INTERACTS_WITH]-(d:Drug {name: $drug})
WHERE t.start <= date($as_of)
  AND (t.stop IS NULL OR t.stop > date($as_of))
RETURN other.name AS drug, t.start AS start, t.stop AS stop,
       collect(i.source) AS sources
ORDER BY drug
"""


def check_exists(driver: Driver, patient_id: str, drug: str) -> None:
    """Raise ValueError if the patient or the drug is not in the graph."""
    records, _, _ = driver.execute_query(
        EXISTS_QUERY, patient_id=patient_id, drug=drug
    )
    if not records[0]["patient_found"]:
        raise ValueError(f"unknown patient: {patient_id}")
    if not records[0]["drug_found"]:
        raise ValueError(f"unknown drug: {drug}")


def iso(value) -> str | None:
    """Turn a Neo4j date into 'YYYY-MM-DD' text; keep None as None."""
    return None if value is None else value.iso_format()


def patient_drug_safety(driver: Driver, patient_id: str, drug: str,
                        as_of: date) -> dict:
    """Return the contraindications and interactions active on as_of."""
    check_exists(driver, patient_id, drug)
    params = {"patient_id": patient_id, "drug": drug, "as_of": as_of.isoformat()}
    conditions, _, _ = driver.execute_query(CONTRAINDICATION_QUERY, **params)
    interactions, _, _ = driver.execute_query(INTERACTION_QUERY, **params)
    contraindications = [
        {"condition": r["condition"], "onset": iso(r["onset"]),
         "resolved": iso(r["resolved"]), "source": r["source"]}
        for r in conditions
    ]
    interacting = [
        {"drug": r["drug"], "start": iso(r["start"]), "stop": iso(r["stop"]),
         "sources": r["sources"]}
        for r in interactions
    ]
    return {
        "patient_id": patient_id,
        "drug": drug,
        "as_of": as_of.isoformat(),
        "contraindications": contraindications,
        "interactions": interacting,
        # WHY "flagged", not "safe": an empty result only means the graph has
        # no matching edge. It is not a statement that the drug is safe.
        "flagged": bool(contraindications or interacting),
    }


if __name__ == "__main__":
    from graph.connect import get_driver

    with get_driver() as driver:
        for patient in ("TEST_ACTIVE", "TEST_RESOLVED"):
            result = patient_drug_safety(
                driver, patient, "spironolactone", date(2026, 6, 1)
            )
            print(patient, "flagged:", result["flagged"], result["contraindications"])
