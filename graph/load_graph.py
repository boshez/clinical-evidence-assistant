"""Load drugs, conditions and the approved edges into Neo4j."""
import csv
from pathlib import Path

from neo4j import Driver

from graph.connect import get_driver
from ingest.chunk import LABEL_DIR, load_label

SCHEMA_FILE = Path("graph/schema.cypher")
CONDITIONS_CSV = Path("data/conditions.csv")
APPROVED_CSV = Path("data/edges_approved.csv")

# WHY: Cypher cannot take a relationship type as a parameter, so each type has
# its own fixed query. Nothing from the CSV is ever pasted into query text.
# MERGE (not CREATE) makes every load safe to run twice.
EDGE_QUERIES = {
    "TREATS": """
        UNWIND $rows AS row
        MATCH (d:Drug {name: row.drug}), (c:Condition {name: row.target})
        MERGE (d)-[r:TREATS]->(c)
        SET r.source = row.source
    """,
    "CONTRAINDICATED_IN": """
        UNWIND $rows AS row
        MATCH (d:Drug {name: row.drug}), (c:Condition {name: row.target})
        MERGE (d)-[r:CONTRAINDICATED_IN]->(c)
        SET r.source = row.source
    """,
    "INTERACTS_WITH": """
        UNWIND $rows AS row
        MATCH (a:Drug {name: row.drug}), (b:Drug {name: row.target})
        MERGE (a)-[r:INTERACTS_WITH]->(b)
        SET r.source = row.source
    """,
}


def run_schema(driver: Driver) -> None:
    """Create the uniqueness constraints listed in schema.cypher."""
    for statement in SCHEMA_FILE.read_text().split(";"):
        if "CREATE" in statement:  # skip the empty piece after the last ;
            driver.execute_query(statement)


def drug_rows() -> list[dict]:
    """One row per saved label, with the properties a Drug node carries."""
    rows = []
    for path in sorted(LABEL_DIR.glob("*.json")):
        label = load_label(path)
        rows.append({
            "name": path.stem,
            "generic_name": label["openfda"]["generic_name"][0],
            "product_type": label["openfda"]["product_type"][0],
            "set_id": label["set_id"],
        })
    return rows


def load_drugs(driver: Driver) -> None:
    """Create or update one Drug node per label."""
    driver.execute_query(
        """
        UNWIND $rows AS row
        MERGE (d:Drug {name: row.name})
        SET d.generic_name = row.generic_name,
            d.product_type = row.product_type,
            d.set_id = row.set_id
        """,
        rows=drug_rows(),
    )


def load_conditions(driver: Driver) -> None:
    """Create one Condition node per canonical name in conditions.csv."""
    with CONDITIONS_CSV.open() as f:
        names = [row["canonical"] for row in csv.DictReader(f)]
    driver.execute_query(
        "UNWIND $names AS name MERGE (:Condition {name: name})", names=names
    )


def approved_rows() -> list[dict]:
    """Only the candidate edges marked y."""
    # WHY: a rejected or unreviewed row must never become a drug fact.
    with APPROVED_CSV.open() as f:
        return [row for row in csv.DictReader(f) if row["approved"] == "y"]


def load_edges(driver: Driver) -> None:
    """Create the approved edges, one batch per relationship type."""
    rows = approved_rows()
    for relation, query in EDGE_QUERIES.items():
        batch = [r for r in rows if r["relation"] == relation]
        driver.execute_query(query, rows=batch)


def count_everything(driver: Driver) -> dict[str, int]:
    """Count nodes per label and relationships per type, to check the load."""
    nodes, _, _ = driver.execute_query(
        "MATCH (n) RETURN labels(n)[0] AS kind, count(*) AS total"
    )
    edges, _, _ = driver.execute_query(
        "MATCH ()-[r]->() RETURN type(r) AS kind, count(*) AS total"
    )
    return {record["kind"]: record["total"] for record in nodes + edges}


if __name__ == "__main__":
    with get_driver() as driver:
        run_schema(driver)
        load_drugs(driver)
        load_conditions(driver)
        load_edges(driver)
        for kind, total in sorted(count_everything(driver).items()):
            print(f"{kind}: {total}")
