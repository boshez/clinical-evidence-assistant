"""Run every gold question through the agent and score answer-or-refuse.

Run with:  python -m eval.agent_eval
It saves after each question and skips questions already done, so it can be
stopped and restarted. Delete eval/results/agent_runs.jsonl to start again.

It measures whether the agent ANSWERED or REFUSED correctly. It does not
judge whether an answer is right; faithfulness.py checks support in the text.
"""
import json
import time

from agent.run import run_agent
from eval.gold import RESULTS_DIR, load_gold
from eval.llm import agent_llm

RUNS = RESULTS_DIR / "agent_runs.jsonl"


def load_runs() -> list[dict]:
    """Results saved so far (one JSON object per line)."""
    if not RUNS.exists():
        return []
    return [json.loads(line) for line in RUNS.read_text().splitlines()]


def run_row(row: dict) -> dict:
    """Ask the agent one gold question and keep what the scoring needs."""
    started = time.time()
    state = run_agent(row["question"], agent_llm)
    chunks = state.get("chunks", [])
    gold = set(row["gold_chunk_ids"].split("|"))
    return {
        "id": row["id"],
        "question": row["question"],
        "answerable": row["answerable"],
        "route": state["route"],
        "attempts": state.get("attempts", 0),
        "final_query": state.get("query", ""),
        "answered": state["confidence"] == "ok",
        "answer": state["answer"],
        # WHY: did the gold chunk reach the answer writer, after any rewrites?
        "gold_in_context": any(c["id"] in gold for c in chunks),
        "chunks": [{"id": c["id"], "text": c["text"]} for c in chunks],
        "seconds": round(time.time() - started, 1),
    }


def share(part: list, whole: list) -> str:
    """'24/32 (0.750)'."""
    return f"{len(part)}/{len(whole)} ({len(part) / len(whole):.3f})"


def report(runs: list[dict]) -> None:
    """Print the answer-or-refuse scores."""
    can = [r for r in runs if r["answerable"] == "y"]
    cannot = [r for r in runs if r["answerable"] == "n"]
    answered = [r for r in can if r["answered"]]
    print(f"answerable, answered:       {share(answered, can)}")
    print(f"answerable, wrongly refused: {', '.join(r['id'] for r in can if not r['answered']) or 'none'}")
    print(f"unanswerable, refused:      {share([r for r in cannot if not r['answered']], cannot)}")
    print(f"unanswerable, wrongly answered: {', '.join(r['id'] for r in cannot if r['answered']) or 'none'}")
    print(f"answered with gold chunk in context: "
          f"{share([r for r in answered if r['gold_in_context']], answered)}")
    print(f"not sent to the label search: {', '.join(r['id'] for r in runs if r['route'] != 'guideline') or 'none'}")
    retried = [r for r in runs if r["attempts"] > 1]
    print(f"needed a rewrite: {len(retried)}/{len(runs)};  "
          f"mean seconds per question: {sum(r['seconds'] for r in runs) / len(runs):.1f}")


if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    done = {r["id"] for r in load_runs()}
    for row in load_gold():
        if row["id"] in done:
            continue
        result = run_row(row)
        with RUNS.open("a") as f:
            f.write(json.dumps(result) + "\n")
        print(f"{result['id']}  {'answered' if result['answered'] else 'refused '}"
              f"  attempts {result['attempts']}  {result['seconds']}s")
    print()
    report(load_runs())
