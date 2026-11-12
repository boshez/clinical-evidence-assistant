"""Open a Neo4j connection using the credentials in .env."""
import os

from dotenv import load_dotenv
from neo4j import Driver, GraphDatabase


def get_driver() -> Driver:
    """Build a Neo4j driver from the three NEO4J_* values in .env."""
    # WHY: credentials live in .env (git-ignored), never in code.
    load_dotenv()
    return GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )


if __name__ == "__main__":
    # WHY: a tiny query proves the URI, username and password all work.
    with get_driver() as driver:
        driver.verify_connectivity()
        records, _, _ = driver.execute_query("MATCH (n) RETURN count(n) AS nodes")
        print(f"connected; nodes in graph: {records[0]['nodes']}")
