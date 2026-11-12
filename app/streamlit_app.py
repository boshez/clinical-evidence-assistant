"""Streamlit UI for the Clinical Evidence Assistant.

Run from the project folder with:  streamlit run app/streamlit_app.py
"""
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


st.set_page_config(page_title="Clinical Evidence Assistant", layout="wide")
st.title("Clinical Evidence Assistant")
st.caption(
    "A portfolio demo, not medical advice. Answers come from 12 openFDA drug "
    "labels and a graph of synthetic patients. It refuses when the labels do "
    "not contain the answer."
)

with st.form("question_form"):
    typed = st.text_input("Ask a question", placeholder=EXAMPLES[0])
    submitted = st.form_submit_button("Ask")
if submitted and typed.strip():
    ask(typed.strip())

st.caption("Or try an example:")
for example in EXAMPLES:
    if st.button(example):
        ask(example)

result = st.session_state.get("result")
if result:
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
