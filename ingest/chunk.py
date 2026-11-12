"""Split saved drug labels into section-aware chunks with metadata."""
import json
import re
from pathlib import Path

LABEL_DIR = Path("data/labels")
# WHY: the embedding model only reads a short input, so chunks must stay small.
MAX_WORDS = 100

# WHY: only top-level clinical sections. Sub-fields such as "pregnancy" repeat
# text from these, and packaging or table fields are noise for retrieval.
SECTIONS = [
    "boxed_warning", "indications_and_usage", "dosage_and_administration",
    "contraindications", "warnings", "warnings_and_cautions", "precautions",
    "drug_interactions", "adverse_reactions", "use_in_specific_populations",
    "overdosage", "clinical_pharmacology",
]


def load_label(path: Path) -> dict:
    """Read one saved label."""
    return json.loads(path.read_text())


def split_sentences(text: str) -> list[str]:
    """Split wherever whitespace follows a full stop, ! or ?."""
    return re.split(r"(?<=[.!?])\s+", text.strip())


def split_long(sentence: str, max_words: int) -> list[str]:
    """Cut an over-long sentence into pieces of at most max_words."""
    # WHY: some label "sentences" are long lists with no full stop.
    words = sentence.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


def count_words(pieces: list[str]) -> int:
    """Total words across a list of text pieces."""
    return sum(len(p.split()) for p in pieces)


def pack_sentences(sentences: list[str], max_words: int = MAX_WORDS) -> list[str]:
    """Group whole sentences into chunks of at most max_words."""
    pieces = [p for s in sentences for p in split_long(s, max_words)]
    chunks: list[str] = []
    current: list[str] = []
    for piece in pieces:
        if current and count_words(current + [piece]) > max_words:
            chunks.append(" ".join(current))
            # WHY: repeating the last sentence keeps context across the cut.
            current = current[-1:]
            if count_words(current + [piece]) > max_words:
                current = []  # the overlap would break the limit, so drop it
        current.append(piece)
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_label(drug: str, label: dict) -> list[dict]:
    """Turn one label into chunks, each tagged with its drug and section."""
    info = label["openfda"]
    chunks = []
    for section in SECTIONS:
        if section not in label:
            continue
        # WHY: older-format labels repeat drug_interactions inside precautions.
        if section == "drug_interactions" and "precautions" in label:
            continue
        text = " ".join(label[section])
        for i, body in enumerate(pack_sentences(split_sentences(text))):
            chunks.append({
                "id": f"{drug}-{section}-{i}",
                "text": body,
                "metadata": {
                    "drug": drug,
                    "generic_name": info["generic_name"][0],
                    "product_type": info["product_type"][0],
                    "section": section,
                    "chunk_index": i,
                    "set_id": label["set_id"],
                },
            })
    return chunks


if __name__ == "__main__":
    total = 0
    for path in sorted(LABEL_DIR.glob("*.json")):
        chunks = chunk_label(path.stem, load_label(path))
        longest = max(len(c["text"].split()) for c in chunks)
        total += len(chunks)
        print(f"{path.stem}: {len(chunks)} chunks, longest {longest} words")
    print(f"total: {total} chunks")