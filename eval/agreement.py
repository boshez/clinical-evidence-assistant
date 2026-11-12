"""Compare the hand check with the judge model.

Run with:  python -m eval.agreement
First fill in human_faithful (y or n) in eval/results/hand_check.csv:
y means every claim in the answer is stated in the context shown beside it.
"""
import csv

from eval.faithfulness import HAND_CHECK, load_judged

if __name__ == "__main__":
    judge = {j["id"]: j["faithful"] for j in load_judged()}
    with HAND_CHECK.open(newline="") as f:
        marked = [r for r in csv.DictReader(f) if r["human_faithful"] in ("y", "n")]
    if not marked:
        raise SystemExit("No rows are marked yet. Fill in human_faithful first.")
    agree = [r for r in marked if (r["human_faithful"] == "y") == judge[r["id"]]]
    print(f"rows marked: {len(marked)}")
    print(f"judge agrees with hand check: {len(agree)}/{len(marked)}")
    for r in marked:
        if r not in agree:
            print(f"  disagree  {r['id']}: human {r['human_faithful']}, "
                  f"judge {'y' if judge[r['id']] else 'n'}")
