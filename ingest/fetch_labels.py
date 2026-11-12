"""Fetch one openFDA drug label per starter drug and save the raw JSON."""
import json
from pathlib import Path

import requests

OPENFDA_URL = "https://api.fda.gov/drug/label.json"
DRUGS = [
    "warfarin", "aspirin", "clopidogrel", "ibuprofen", "metformin", "lisinopril",
    "spironolactone", "furosemide", "metoprolol", "amlodipine", "atorvastatin",
    "omeprazole",
]

RX = "HUMAN PRESCRIPTION DRUG"
OTC = "HUMAN OTC DRUG"
# WHY: a label missing these sections cannot answer safety questions.
REQUIRED = {
    RX: ["indications_and_usage", "contraindications", "drug_interactions"],
    OTC: ["indications_and_usage", "warnings", "do_not_use"],
}
OUT_DIR = Path("data/labels")


def fetch_candidates(drug: str, product_type: str, limit: int = 100) -> list[dict]:
    """Return up to `limit` labels of one product type for `drug`."""
    search = (
        f'openfda.generic_name:"{drug}" '
        f'AND openfda.product_type:"{product_type}"'
    )
    resp = requests.get(
        OPENFDA_URL, params={"search": search, "limit": limit}, timeout=60
    )
    if resp.status_code == 404:  # WHY: openFDA answers 404 when nothing matches
        return []
    resp.raise_for_status()
    return resp.json()["results"]


def is_single_ingredient(label: dict) -> bool:
    """True when the label covers exactly one active substance."""
    # WHY: a combination product would mix another drug's facts into this one.
    return len(label.get("openfda", {}).get("substance_name", [])) == 1


def pick_label(candidates: list[dict], required: list[str]) -> dict | None:
    """Pick the newest single-ingredient label that has every required section."""
    usable = [
        c for c in candidates
        if is_single_ingredient(c) and all(s in c for s in required)
    ]
    # WHY: effective_time is YYYYMMDD text, so the largest string is the newest.
    return max(usable, key=lambda c: c["effective_time"], default=None)


def get_label(drug: str) -> dict | None:
    """Prefer a prescription label; fall back to over-the-counter."""
    # WHY: prescription labels have the full clinical sections. Only aspirin
    # has no single-ingredient prescription label, so it needs the fallback.
    for product_type in (RX, OTC):
        candidates = fetch_candidates(drug, product_type)
        label = pick_label(candidates, REQUIRED[product_type])
        if label is not None:
            return label
    return None


def save_label(drug: str, label: dict) -> Path:
    """Write the raw label to data/labels/<drug>.json."""
    # WHY: saving raw JSON means chunking can be re-run without re-fetching.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{drug}.json"
    path.write_text(json.dumps(label, indent=2))
    return path


if __name__ == "__main__":
    for drug in DRUGS:
        label = get_label(drug)
        if label is None:
            print(f"{drug}: NO USABLE LABEL")
            continue
        path = save_label(drug, label)
        info = label["openfda"]
        print(
            f"{drug}: {info['generic_name'][0]} | {info['product_type'][0]} "
            f"| {label['effective_time']} | {path}"
        )
