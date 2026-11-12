"""Check the gold set against the indexed label text and fill gold_chunk_ids.

Run with:  python -m eval.check_gold

WHY: a gold answer is only worth anything if its supporting sentence is really
in the label. This proves each quote is there word for word, and that each
"unanswerable" topic is absent. It does NOT prove the answer is a fair reading
of the quote; that is the hand check recorded in the verified column.
"""
from eval.gold import GOLD_CSV, chunks_for, load_gold, save_rows


def squash(text: str) -> str:
    """Lower-case and collapse whitespace, so spacing cannot break a match."""
    return " ".join(text.lower().split())


def check_answerable(row: dict) -> str:
    """Fill gold_chunk_ids; return an error message or ''."""
    quote = squash(row["source_quote"])
    hits = [c for c in chunks_for(row["drug"]) if quote in squash(c["text"])]
    # WHY several ids: chunks overlap by one sentence, so a quote can sit in two.
    row["gold_chunk_ids"] = "|".join(sorted(c["id"] for c in hits))
    if not hits:
        return "quote not found in any chunk"
    if row["section"] not in {c["section"] for c in hits}:
        return f"quote is not in section {row['section']}"
    return ""


def check_unanswerable(row: dict) -> str:
    """Confirm none of the absent_terms appear in the drug's chunks."""
    text = squash(" ".join(c["text"] for c in chunks_for(row["drug"])))
    found = [t for t in row["absent_terms"].split("|") if squash(t) in text]
    return f"term present in label: {found}" if found else ""


if __name__ == "__main__":
    rows = load_gold()
    problems = 0
    for row in rows:
        check = check_answerable if row["answerable"] == "y" else check_unanswerable
        error = check(row)
        if error:
            problems += 1
            print(f"FAIL  {row['id']}  {error}")
    save_rows(GOLD_CSV, rows)
    verified = sum(r["verified"] == "y" for r in rows)
    print(f"{len(rows)} rows checked, {problems} problems")
    print(f"hand-verified so far: {verified}/{len(rows)}")
