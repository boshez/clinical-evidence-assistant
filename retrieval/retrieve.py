"""Search the Chroma index of drug-label chunks."""
from functools import lru_cache

import chromadb
from sentence_transformers import SentenceTransformer

from ingest.index import CHROMA_DIR, COLLECTION, MODEL_NAME


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Load the embedding model once and reuse it."""
    # WHY: loading takes seconds, and the agent will call retrieve many times.
    return SentenceTransformer(MODEL_NAME)


@lru_cache(maxsize=1)
def get_collection() -> chromadb.Collection:
    """Open the saved Chroma collection once and reuse it."""
    return chromadb.PersistentClient(path=CHROMA_DIR).get_collection(COLLECTION)


def retrieve(query: str, k: int = 5) -> list[dict]:
    """Return the k chunks closest to the query, best first."""
    embedding = get_model().encode([query]).tolist()
    result = get_collection().query(query_embeddings=embedding, n_results=k)
    rows = zip(
        result["ids"][0], result["documents"][0],
        result["metadatas"][0], result["distances"][0],
    )
    # WHY: Chroma returns cosine distance; 1 - distance is similarity (higher = better).
    return [
        {"id": i, "text": t, "metadata": m, "score": round(1 - d, 3)}
        for i, t, m, d in rows
    ]