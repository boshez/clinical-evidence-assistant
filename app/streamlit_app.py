"""Streamlit UI for the Clinical Evidence Assistant.

Run from the project folder with:  streamlit run app/streamlit_app.py
"""
import csv
import os
import sys
from pathlib import Path

# WHY: Streamlit puts app/ on the import path, not the project folder, and
# the agent reads chroma/ and data/ by relative path. Fix both before imports.
ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from agent.run import run_agent  # noqa: E402
from eval.agent_eval import load_runs  # noqa: E402
from eval.faithfulness import load_judged  # noqa: E402
from eval.gold import RESULTS_DIR, load_gold  # noqa: E402
from ingest.fetch_labels import DRUGS  # noqa: E402

EXAMPLES = [
    "Who should not take metformin?",
    "Can patient TEST_RESOLVED take spironolactone on 2026-06-01?",
    "Can patient TEST_TAKES_NOW start ibuprofen on 2026-06-01?",
    "What is the readmission risk for patient P00017?",
    "Does metformin interact with grapefruit juice?",
]
ROUTE_NAMES = {
    "guideline": "drug-label search",
    "safety": "patient safety check (graph)",
    "risk": "readmission-risk lookup",
    "refuse": "refused before any tool",
}


def plain_hyphens(text: str) -> str:
    """Turn typographic hyphens into '-' so cited chunk ids can be matched."""
    # WHY: the model sometimes writes ids with non-breaking hyphens.
    return text.replace("‑", "-").replace("‐", "-")


def show_answer(state: dict) -> None:
    """The answer, with a clear marker when the agent refused."""
    if state["confidence"] == "ok":
        st.success(f"Answered using: {ROUTE_NAMES[state['route']]}")
    else:
        st.warning("Low confidence: the assistant refused instead of guessing.")
    st.markdown(plain_hyphens(state["answer"]))


def show_label_sources(state: dict) -> None:
    """The retrieved label chunks, cited ones first."""
    answer = plain_hyphens(state["answer"])
    chunks = sorted(state["chunks"], key=lambda c: c["id"] not in answer)
    for chunk in chunks:
        meta = chunk["metadata"]
        cited = "cited" if chunk["id"] in answer else "retrieved, not cited"
        title = f"{chunk['id']}  ({cited}, score {chunk['score']})"
        with st.expander(title, expanded=chunk["id"] in answer):
            st.caption(f"{meta['generic_name']} | {meta['section']} | "
                       f"{meta['product_type']} | openFDA set_id {meta['set_id']}")
            st.write(chunk["text"])


def show_tool_sources(state: dict) -> None:
    """Where a safety or risk answer came from."""
    result = state["tool_result"]
    if state["route"] == "risk":
        st.caption("Source: data/patient_risk_scores.csv (synthetic patients), "
                   f"admission of {result['admission_date']}.")
        return
    st.caption("Source: the knowledge graph. Each flag quotes the label "
               "sentence stored on its edge. Patients are synthetic.")
    st.json(result, expanded=False)


def show_sources(state: dict) -> None:
    """Pick the right kind of source list for the route taken."""
    if state["confidence"] != "ok":
        st.caption("No sources: nothing was answered.")
    elif state["route"] == "guideline":
        show_label_sources(state)
    else:
        show_tool_sources(state)


def show_trace(state: dict) -> None:
    """One line per node that ran, with its details behind a click."""
    for step, record in enumerate(state["trace"], start=1):
        with st.expander(f"{step}. {record['node']}: {record['summary']}"):
            st.caption(record["at"])
            st.json(record["detail"])


def ask(question: str) -> None:
    """Run the agent and keep the result for redraws."""
    with st.spinner("Working through the steps..."):
        try:
            st.session_state["result"] = run_agent(question)
        except Exception as error:  # WHY broad: show any failure, do not crash
            st.session_state["result"] = None
            st.error(f"The assistant could not run: {error}")


def load_test_results() -> list[dict]:
    """Join each gold question with what the agent did on the saved run."""
    runs = {r["id"]: r for r in load_runs()}
    rows = []
    for gold in load_gold():
        run = runs.get(gold["id"])
        if run is None:
            continue
        should_answer = gold["answerable"] == "y"
        rows.append({
            "id": gold["id"],
            "question": gold["question"],
            "should": "answer" if should_answer else "refuse",
            "did": "answered" if run["answered"] else "refused",
            "correct": should_answer == run["answered"],
            "gold_answer": gold["gold_answer"],
            "source_quote": gold["source_quote"],
            "agent_answer": run["answer"],
            "verified": gold["verified"] == "y",
        })
    return rows


def show_scores(rows: list[dict]) -> None:
    """The headline numbers, computed from the saved result files."""
    can = [r for r in rows if r["should"] == "answer"]
    cannot = [r for r in rows if r["should"] == "refuse"]
    judged = [j for j in load_judged() if j["n_claims"]]
    with (RESULTS_DIR / "retrieval.csv").open(newline="") as f:
        ranks = [int(r["strict_rank"]) for r in csv.DictReader(f)]
    one, two, three, four = st.columns(4)
    one.metric("Answerable questions answered",
               f"{sum(r['correct'] for r in can)} / {len(can)}")
    two.metric("Unanswerable questions refused",
               f"{sum(r['correct'] for r in cannot)} / {len(cannot)}")
    three.metric("Claims supported by the label text",
                 f"{sum(j['n_supported'] for j in judged)} / "
                 f"{sum(j['n_claims'] for j in judged)}")
    four.metric("Search finds the exact passage in top 5",
                f"{sum(0 < r <= 5 for r in ranks)} / {len(ranks)}")


def show_tests() -> None:
    """The evaluation: scores, limits, and every test question."""
    rows = load_test_results()
    show_scores(rows)
    st.caption(
        "One run of 40 questions written from the label text; 8 have no answer "
        "in the labels on purpose. Claim support was marked by a second "
        "language model. These numbers show no failures on this set, not zero "
        "error. Search alone misses some passages; the rewrite-and-retry step "
        f"recovers most of them. Gold answers hand-verified so far: "
        f"{sum(r['verified'] for r in rows)} of {len(rows)}."
    )
    st.dataframe(
        [{"id": r["id"], "question": r["question"], "should": r["should"],
          "did": r["did"], "correct": "yes" if r["correct"] else "NO"}
         for r in rows],
        hide_index=True, width="stretch",
    )
    labels = {f"{r['id']}  {r['question']}": r for r in rows}
    row = labels[st.selectbox("Look at one test question", list(labels))]
    if row["should"] == "answer":
        st.markdown(f"**Expected answer:** {row['gold_answer']}")
        st.markdown(f"**Label sentence behind it:** \"{row['source_quote']}\"")
    else:
        st.markdown("**Expected:** refuse, because the label does not cover this.")
    st.markdown("**What the assistant said:**")
    st.info(plain_hyphens(row["agent_answer"]))


def show_sidebar() -> None:
    """What the assistant is and what it can be asked."""
    st.sidebar.header("What you can ask")
    st.sidebar.markdown(
        "**1. What a drug label says**\n"
        "uses, who should not take it, dose, interactions, side effects, "
        "overdose.\n\n"
        "**2. Is a drug a problem for a patient on a date?**\n"
        "give a patient ID, a drug and a date.\n\n"
        "**3. A patient's readmission risk**\n"
        "give a patient ID.\n\n"
        f"**The 12 drugs:** {', '.join(DRUGS)}.\n\n"
        "**Test patients:** TEST_ACTIVE, TEST_RESOLVED, TEST_FUTURE, "
        "TEST_BOUNDARY, TEST_TAKES_NOW.\n\n"
        "It refuses anything the labels do not state. That is deliberate: "
        "a wrong answer is worse than no answer."
    )
    st.sidebar.caption("Answers take about 20 seconds on the free model tier.")


def show_ask() -> None:
    """The question box, the ready-made questions, and the result."""
    ready = [""] + EXAMPLES + [r["question"] for r in load_gold()]
    with st.form("question_form"):
        typed = st.text_input("Type any question", placeholder=EXAMPLES[0])
        picked = st.selectbox(
            "or pick a ready-made one (5 examples, then the 40 test questions)",
            ready,
        )
        submitted = st.form_submit_button("Ask")
    question = typed.strip() or picked
    if submitted and question:
        ask(question)

    result = st.session_state.get("result")
    if not result:
        return
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Answer")
        st.markdown(f"**Question:** {result['question']}")
        show_answer(result)
        st.subheader("Sources")
        show_sources(result)
    with right:
        st.subheader("Reasoning trace")
        show_trace(result)


st.set_page_config(page_title="Clinical Evidence Assistant", layout="wide")
st.title("Clinical Evidence Assistant")
st.caption(
    "A portfolio demo, not medical advice. Answers come from 12 openFDA drug "
    "labels and a graph of synthetic patients. It refuses when the labels do "
    "not contain the answer."
)
show_sidebar()
ask_tab, tests_tab = st.tabs(["Ask a question", "How it was tested"])
with ask_tab:
    show_ask()
with tests_tab:
    show_tests()
