"""Tiny check that retrieval finds the right drug and section."""
from retrieval.retrieve import retrieve

# (question, expected drug, expected section)
CASES = [
    ("Can warfarin be used during pregnancy?", "warfarin", "contraindications"),
    ("What is metformin used for?", "metformin", "indications_and_usage"),
    ("Which drugs interact with clopidogrel?", "clopidogrel", "drug_interactions"),
]


def hit(question: str, drug: str, section: str, k: int = 5) -> bool:
    """True if any of the top k chunks comes from the expected drug and section."""
    chunks = retrieve(question, k)
    found = [(c["metadata"]["drug"], c["metadata"]["section"]) for c in chunks]
    return (drug, section) in found


if __name__ == "__main__":
    results = [hit(*case) for case in CASES]
    for case, ok in zip(CASES, results):
        print("PASS" if ok else "FAIL", "-", case[0])
    print(f"{sum(results)}/{len(CASES)} passed")