"""Faithfulness: is every claim in an answer supported by the retrieved text?

Run with:  python -m eval.faithfulness   (after python -m eval.agent_eval)

A second model (the judge) lists the factual claims in each answer and marks
each one supported or not, using ONLY the chunks the agent retrieved.
Faithful is not the same as correct: an answer can be fully supported by the
wrong chunk. Only answers the agent actually gave are judged.
"""
import json
import random

from agent.llm import parse_json
from eval.agent_eval import load_runs
from eval.gold import RESULTS_DIR, save_rows
from eval.llm import JUDGE_MODEL, judge_llm

JUDGED = RESULTS_DIR / "faithfulness.jsonl"
HAND_CHECK = RESULTS_DIR / "hand_check.csv"
HAND_CHECK_SIZE = 10

JUDGE_PROMPT = """You check an answer against source text from drug labels.
1. List every factual claim the answer makes (split lists into single claims).
2. For each claim, decide whether the CONTEXT states it. Use only the context,
   not your own knowledge. A claim the context does not state is unsupported,
   even if it is true.
Reply with one JSON object and nothing else:
{"claims": [{"claim": "...", "supported": true or false}]}"""


def context(chunks: list[dict]) -> str:
    """The retrieved chunks as the judge sees them."""
    return "\n\n".join(f"[{c['id']}]\n{c['text']}" for c in chunks)


def judge(run: dict) -> dict:
    """Ask the judge about one answer; return its claims and the score."""
    reply = parse_json(judge_llm(
        JUDGE_PROMPT,
        f"ANSWER:\n{run['answer']}\n\nCONTEXT:\n{context(run['chunks'])}",
    ))
    claims = [c for c in reply.get("claims", []) if isinstance(c, dict)]
    supported = sum(c.get("supported") is True for c in claims)
    return {
        "id": run["id"],
        "claims": claims,
        "n_claims": len(claims),
        "n_supported": supported,
        # WHY: no readable claims means the judge failed, not that the
        # answer is faithful. It is counted separately.
        "faithful": bool(claims) and supported == len(claims),
    }


def load_judged() -> list[dict]:
    """Judge results saved so far."""
    if not JUDGED.exists():
        return []
    return [json.loads(line) for line in JUDGED.read_text().splitlines()]


def write_hand_check(runs: list[dict]) -> None:
    """Write a sheet of answers for a person to mark, without the judge's verdict."""
    # WHY no verdict in the sheet: seeing it would bias the person marking.
    if HAND_CHECK.exists():
        return  # never overwrite marks that have been filled in
    sample = random.Random(42).sample(runs, min(HAND_CHECK_SIZE, len(runs)))
    save_rows(HAND_CHECK, [{
        "id": r["id"], "question": r["question"], "answer": r["answer"],
        "context": context(r["chunks"]), "human_faithful": "",
    } for r in sorted(sample, key=lambda r: r["id"])])


def report(judged: list[dict]) -> None:
    """Print claim-level and answer-level faithfulness."""
    readable = [j for j in judged if j["n_claims"]]
    claims = sum(j["n_claims"] for j in readable)
    supported = sum(j["n_supported"] for j in readable)
    faithful = [j for j in readable if j["faithful"]]
    print(f"judge model: {JUDGE_MODEL}")
    print(f"answers judged: {len(readable)} (judge reply unreadable: {len(judged) - len(readable)})")
    print(f"claims supported: {supported}/{claims} ({supported / claims:.3f})")
    print(f"answers with every claim supported: {len(faithful)}/{len(readable)} "
          f"({len(faithful) / len(readable):.3f})")
    for j in readable:
        for c in j["claims"]:
            if c.get("supported") is not True:
                print(f"  unsupported  {j['id']}: {c.get('claim')}")


if __name__ == "__main__":
    answered = [r for r in load_runs() if r["answered"] and r["chunks"]]
    done = {j["id"] for j in load_judged()}
    for run in answered:
        if run["id"] in done:
            continue
        with JUDGED.open("a") as f:
            f.write(json.dumps(judge(run)) + "\n")
    write_hand_check(answered)
    report(load_judged())
    print(f"\nhand-check sheet: {HAND_CHECK} (fill in human_faithful with y or n)")
