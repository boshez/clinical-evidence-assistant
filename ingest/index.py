"""Embed the label chunks and store them in a persistent Chroma index."""
import chromadb
from sentence_transformers import SentenceTransformer

from ingest.chunk import LABEL_DIR, chunk_label, load_label

MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_DIR = "chroma"
COLLECTION = "drug_labels"

def load_chunks() -> list[dict]:
    """Chunk every saved label."""
    chunks = []
    for path in sorted(LABEL_DIR.glob("*.json")):
        chunks.extend(chunk_label(path.stem, load_label(path)))
    return chunks


def embedding_text(chunk: dict) -> str:
    """The text we embed: the chunk with its drug and section in front."""
    # WHY: many chunks never name their drug, so the prefix tells the model.
    meta = chunk["metadata"]
    section = meta["section"].replace("_", " ")
    return f"{meta['generic_name']} - {section}: {chunk['text']}"
    
def fresh_collection(client: chromadb.ClientAPI) -> chromadb.Collection:
    """Delete any old collection and create an empty one."""
    # WHY: rebuilding from scratch means no stale or duplicate chunks survive.
    if COLLECTION in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION)
    # WHY: cosine distance compares direction, the usual choice for text embeddings.
    return client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    
def build_index() -> int:
    """Embed all chunks, store them, and return how many were stored."""
    model = SentenceTransformer(MODEL_NAME)
    print(f"model input limit: {model.max_seq_length} tokens")
    chunks = load_chunks()
    texts = [embedding_text(c) for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    collection = fresh_collection(chromadb.PersistentClient(path=CHROMA_DIR))
    collection.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
        embeddings=embeddings.tolist(),
    )
    return collection.count()


if __name__ == "__main__":
    print(f"indexed {build_index()} chunks")