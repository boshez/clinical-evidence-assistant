"""Propose candidate graph edges from label text, for a human to approve.

Nothing here decides a drug fact. It only finds label sentences that mention a
known condition or drug, and writes them to a CSV with an empty `approved` column.
"""
import csv
import re
from pathlib import Path

from ingest.chunk import LABEL_DIR, load_label, split_sentences

CONDITIONS_CSV = Path("data/conditions.csv")
CANDIDATES_CSV = Path("data/edges_candidates.csv")

# WHY: the label section decides the relationship type, not the wording.
# Aspirin's over-the-counter label has no contraindications or drug_interactions
# field, so its nearest equivalents are listed too.
CONDITION_SECTIONS = {
    "indications_and_usage": "TREATS",
    "contraindications": "CONTRAINDICATED_IN",
    "do_not_use": "CONTRAINDICATED_IN",
}
INTERACTION_SECTIONS = ["drug_interactions", "ask_doctor_or_pharmacist"]

# WHY: labels often name a drug class ("NSAIDs") instead of a specific drug, so
# each of our 12 drugs is matched by its name and its standard class names.
DRUG_TERMS = {
    "warfarin": ["warfarin", "anticoagulant"],
    "aspirin": ["aspirin", "nsaid", "antiplatelet"],
    "clopidogrel": ["clopidogrel", "antiplatelet"],
    "ibuprofen": ["ibuprofen", "nsaid", "nonsteroidal anti-inflammatory"],
    "metformin": ["metformin"],
    "lisinopril": ["lisinopril", "ace inhibitor",
                   "angiotensin converting enzyme inhibitor"],
    "spironolactone": ["spironolactone", "potassium-sparing diuretic", "diuretic"],
    "furosemide": ["furosemide", "loop diuretic", "diuretic"],
    "metoprolol": ["metoprolol", "beta-blocker", "beta blocker"],
    "amlodipine": ["amlodipine", "calcium channel blocker"],
    "atorvastatin": ["atorvastatin", "statin", "hmg-coa reductase inhibitor"],
    "omeprazole": ["omeprazole", "proton pump inhibitor", "ppi"],
}


def load_condition_aliases() -> list[tuple[str, str]]:
    """Return (alias, canonical) pairs, longest alias first."""
    pairs = []
    with CONDITIONS_CSV.open() as f:
        for row in csv.DictReader(f):
            for alias in row["aliases"].split("|"):
                pairs.append((alias, row["canonical"]))
    # WHY: longest first, so "decompensated heart failure" is matched before
    # the shorter "heart failure" hidden inside it.
    return sorted(pairs, key=lambda p: len(p[0]), reverse=True)


def normalise(text: str) -> str:
    """Lower-case the text and straighten curly apostrophes and odd hyphens."""
    # WHY: labels write "Addison’s" and "non‑steroidal" with typographic characters.
    text = text.lower().replace("’", "'").replace("‑", "-").replace("–", "-")
    return text.replace("non-steroidal", "nonsteroidal")


def term_pattern(term: str) -> re.Pattern:
    """Match a whole term, allowing a plural s, but not inside a longer word."""
    # WHY: without the boundaries, "statin" would match inside "atorvastatin".
    return re.compile(r"(?<![a-z])" + re.escape(term) + r"s?(?![a-z])")


def find_conditions(sentence: str, aliases: list[tuple[str, str]]) -> dict[str, str]:
    """Return {canonical condition: matched alias} for one sentence."""
    text = normalise(sentence)
    found: dict[str, str] = {}
    for alias, canonical in aliases:
        pattern = term_pattern(alias)
        if pattern.search(text):
            found.setdefault(canonical, alias)
            # WHY: blank out the match so a shorter alias cannot match it again.
            text = pattern.sub(" ", text)
    return found


def find_drugs(sentence: str, own_drug: str) -> dict[str, str]:
    """Return {other drug: matched term} for one sentence."""
    text = normalise(sentence)
    found: dict[str, str] = {}
    for drug, terms in DRUG_TERMS.items():
        if drug == own_drug:
            continue  # a label mentioning itself is not an interaction
        for term in terms:
            if term_pattern(term).search(text):
                found[drug] = term
                break
    return found


def candidates_for_label(drug: str, label: dict,
                         aliases: list[tuple[str, str]]) -> list[dict]:
    """Every (relation, target, sentence) candidate found in one label."""
    rows = []
    for section, relation in CONDITION_SECTIONS.items():
        for sentence in split_sentences(" ".join(label.get(section, []))):
            for target, term in find_conditions(sentence, aliases).items():
                rows.append(make_row(drug, relation, target, term, section, sentence))
    for section in INTERACTION_SECTIONS:
        for sentence in split_sentences(" ".join(label.get(section, []))):
            for target, term in find_drugs(sentence, drug).items():
                rows.append(
                    make_row(drug, "INTERACTS_WITH", target, term, section, sentence)
                )
    return rows


def make_row(drug: str, relation: str, target: str, term: str,
             section: str, sentence: str) -> dict:
    """One CSV row. `approved` starts empty: a human fills in y or n."""
    return {
        "approved": "", "drug": drug, "relation": relation, "target": target,
        "matched_term": term, "section": section, "source": sentence,
    }


# WHY: a sentence that says "indicated" or "contraindicated" states the
# relationship outright, so it is better evidence than a passing mention.
CUE_WORDS = {"TREATS": "indicated", "CONTRAINDICATED_IN": "contraindicated"}


def source_rank(row: dict) -> tuple[bool, bool, int]:
    """Lower is better: cue word first, then exact name, then shorter sentence."""
    # WHY: a sentence naming "warfarin" is stronger evidence than one that only
    # says "anticoagulants", and a short sentence is quicker for a human to check.
    cue = CUE_WORDS.get(row["relation"], "")  # interactions have no cue word
    return (
        cue not in row["source"].lower(),
        row["matched_term"] != row["target"],
        len(row["source"]),
    )


def keep_best_source(rows: list[dict]) -> list[dict]:
    """Keep one row per (drug, relation, target): the best evidence sentence."""
    best: dict[tuple, dict] = {}
    for row in rows:
        key = (row["drug"], row["relation"], row["target"])
        if key not in best or source_rank(row) < source_rank(best[key]):
            best[key] = row
    return list(best.values())


def write_candidates(rows: list[dict]) -> None:
    """Write the candidate edges to data/edges_candidates.csv."""
    with CANDIDATES_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    aliases = load_condition_aliases()
    all_rows: list[dict] = []
    for path in sorted(LABEL_DIR.glob("*.json")):
        found = candidates_for_label(path.stem, load_label(path), aliases)
        all_rows.extend(keep_best_source(found))
    write_candidates(all_rows)
    for relation in ("TREATS", "CONTRAINDICATED_IN", "INTERACTS_WITH"):
        count = sum(1 for r in all_rows if r["relation"] == relation)
        print(f"{relation}: {count} candidates")
    print(f"wrote {len(all_rows)} rows to {CANDIDATES_CSV}")
