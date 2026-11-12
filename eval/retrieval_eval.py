"""Retrieval metrics on the gold set: Hit Rate@K and MRR. No LLM is used.

Run with:  python -m eval.retrieval_eval

Each answerable question is searched exactly as written (no rewriting), so
this measures the retriever alone. Two definitions of a hit are reported:
  strict: a retrieved chunk is one of the row's gold_chunk_ids
  loose:  a retrieved chunk is from the same drug and label section
"""
from statistics import median

from eval.gold import RESULTS_DIR, answerable, load_gold, save_rows
from retrieval.retrieve import retrieve

K = 5
KS = (1, 3, 5)
# Not part of the gold set. Used only to show what the score threshold is
# for: separating label questions from questions about something else.
OFF_TOPIC = [
    "What is the capital of France?",
    "How do I reset my router password?",
    "Who won the 2018 football World Cup?",
    "What is a good recipe for banana bread?",
    "How do I change a flat tyre?",
]


def section_key(chunk_id: str) -> str:
    """'metformin-contraindications-0' -> 'metformin-contraindications'."""
    return chunk_id.rsplit("-", 1)[0]


def first_hit_rank(retrieved: list[str], gold: set[str]) -> int:
    """1-based position of the first retrieved id in gold; 0 if none."""
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in gold:
            return rank
    return 0


def score_row(row: dict) -> dict:
    """Search one question and record where the gold chunk landed."""
    chunks = retrieve(row["question"], K)
    ids = [c["id"] for c in chunks]
    gold = set(row["gold_chunk_ids"].split("|"))
    return {
        "id": row["id"],
        "question": row["question"],
        "top_score": chunks[0]["score"],
        "strict_rank": first_hit_rank(ids, gold),
        "loose_rank": first_hit_rank(
            [section_key(i) for i in ids], {section_key(g) for g in gold}
        ),
        "retrieved": "|".join(ids),
    }


def hit_rate(ranks: list[int], k: int) -> float:
    """Share of questions whose first hit is in the top k."""
    return sum(0 < r <= k for r in ranks) / len(ranks)


def mrr(ranks: list[int]) -> float:
    """Mean of 1/rank; a question with no hit in the top K counts as 0."""
    return sum(1 / r for r in ranks if r) / len(ranks)


def report(name: str, ranks: list[int]) -> None:
    """Print one line of metrics for one hit definition."""
    rates = "  ".join(f"Hit@{k} {hit_rate(ranks, k):.3f}" for k in KS)
    print(f"{name:<7}{rates}  MRR@{K} {mrr(ranks):.3f}")


def top_scores(questions: list[str]) -> list[float]:
    """The best similarity score for each question."""
    return [retrieve(q, 1)[0]["score"] for q in questions]


def describe(name: str, scores: list[float]) -> None:
    """Print the spread of a list of scores."""
    print(f"{name:<28}n={len(scores):<3} min {min(scores):.3f}  "
          f"median {median(scores):.3f}  max {max(scores):.3f}")


if __name__ == "__main__":
    gold = load_gold()
    rows = [score_row(r) for r in answerable(gold)]
    save_rows(RESULTS_DIR / "retrieval.csv", rows)

    print(f"{len(rows)} answerable questions, searched as written\n")
    report("strict", [r["strict_rank"] for r in rows])
    report("loose", [r["loose_rank"] for r in rows])
    missed = [r["id"] for r in rows if not r["strict_rank"]]
    print(f"\nno gold chunk in top {K}: {', '.join(missed) or 'none'}")

    print("\nTop similarity score by question type")
    describe("answerable", [r["top_score"] for r in rows])
    unanswerable = [r["question"] for r in gold if r["answerable"] == "n"]
    describe("unanswerable (in-domain)", top_scores(unanswerable))
    describe("off-topic (not in gold set)", top_scores(OFF_TOPIC))
