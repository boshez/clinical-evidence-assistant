"""The agent's nodes. Each takes the state and returns only what it changes."""
from datetime import date

from agent import tools
from agent.llm import LLM, parse_json
from agent.state import MAX_ATTEMPTS, SCORE_THRESHOLD, AgentState
from agent.trace import entry

ROUTES = ("guideline", "safety", "risk", "refuse")

ROUTE_PROMPT = """You route questions for a clinical evidence assistant.
Reply with one JSON object and nothing else:
{"route": "...", "patient_id": null, "drug": null, "as_of": null}
route is one of:
- "guideline": a question about what a drug label says (uses, warnings,
  contraindications, interactions, dosing).
- "safety": whether a drug is a problem for ONE NAMED PATIENT (needs a patient ID).
- "risk": a named patient's hospital readmission risk.
- "refuse": not about drugs or these patients.
patient_id: the ID exactly as written (for example P00017), else null.
drug: the generic drug name in lower case, else null.
as_of: a date as YYYY-MM-DD only if the question gives one, else null."""

CRITIQUE_PROMPT = """You check retrieved drug-label text before an answer is written.
Decide whether the context states the information needed to answer the question.
Related text about the same drug is NOT enough: the specific fact must be there.
Each chunk is headed by its label section. The section must match what is asked:
"who should not take it" needs the contraindications section; risk factors,
warnings or precautions do not answer it. If the matching section is missing,
the context is not sufficient.
Reply with one JSON object and nothing else:
{"sufficient": true or false, "reason": "one short sentence"}"""

REWRITE_PROMPT = """A search over FDA drug-label text found nothing useful.
Rewrite the question as a short search query using the words a drug label uses
(for example: indications, contraindications, drug interactions, warnings,
adverse reactions, dosage). Keep the drug name. Do not repeat the previous
query. Reply with the query only."""

ANSWER_PROMPT = """Answer the question using ONLY the context below.
After each fact, cite the chunk it came from in plain square brackets, like
[metformin-contraindications-0]. If a chunk comes from an over-the-counter
label, say so. Do not add anything that is not in the context, and do not
present a risk factor or warning as a contraindication.
If the context does not contain the answer, reply with exactly: INSUFFICIENT"""

OFF_TOPIC = "The question is outside what the drug labels and patient data cover."
# WHY: a bare refusal leaves the user stuck. Broad questions ("what does the
# label say?") fail because only 5 passages are read, so say how to narrow it.
NARROW_HINT = (
    "Try asking about one topic for one drug, such as its uses, dose, "
    "contraindications, interactions or side effects."
)
NOT_SAFE_NOTE = (
    "Nothing was flagged in the graph for this date. That is not a statement "
    "that the drug is safe: the graph holds only 12 drugs and the label "
    "statements that were approved as edges."
)


def low_confidence(reason: str) -> dict:
    """The refusal every dead end uses, so it is always worded the same way."""
    return {
        "answer": f"I cannot answer this with confidence, so I will not guess. {reason}",
        "confidence": "low",
        "reason": reason,
    }


def router(state: AgentState, llm: LLM) -> dict:
    """Pick a tool and pull out the arguments it needs."""
    reply = parse_json(llm(ROUTE_PROMPT, state["question"]))
    route = reply.get("route")
    # WHY default to guideline: if the reply is unreadable, retrieval plus the
    # critique step can still refuse. No other route has that safety net.
    if route not in ROUTES:
        route = "guideline"
    patient_id = clean_patient_id(reply.get("patient_id"))
    drug = reply.get("drug")
    update = {
        "route": route,
        "patient_id": patient_id,
        "drug": drug.lower() if isinstance(drug, str) else None,
        # WHY today: a safety question with no date means "right now".
        "as_of": reply.get("as_of") or date.today().isoformat(),
        "query": state["question"],
        "attempts": 0,
    }
    problem = missing_argument(route, patient_id, drug)
    if problem:
        update.update(route="refuse", reason=problem)
    elif route == "refuse":
        update["reason"] = OFF_TOPIC
    update["trace"] = entry("router", f"route = {update['route']}", **{
        k: update.get(k) for k in ("patient_id", "drug", "as_of", "reason")
    })
    return update


def clean_patient_id(value: object) -> str | None:
    """Tidy a typed patient ID: '163', '00163' or 'p163' becomes 'P00163'."""
    # WHY: IDs in the data are 'P' plus five digits, but people type them
    # loosely. Without this, a real patient is refused as unknown.
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().upper()
    digits = text[1:] if text.startswith("P") else text
    return f"P{int(digits):05d}" if digits.isdigit() else text


def missing_argument(route: str, patient_id: str | None, drug: str | None) -> str:
    """Say what a tool is missing, or '' if it has what it needs."""
    if route == "safety" and not (patient_id and drug):
        return "A safety check needs both a patient ID and a drug name."
    if route == "risk" and not patient_id:
        return "A readmission-risk lookup needs a patient ID."
    return ""


def retrieve(state: AgentState) -> dict:
    """Search the label index with the current query."""
    chunks = tools.guideline_lookup(state["query"])
    attempts = state["attempts"] + 1
    return {
        "chunks": chunks,
        "attempts": attempts,
        "trace": entry(
            "retrieve", f"attempt {attempts}: top score {chunks[0]['score']}",
            query=state["query"],
            hits=[{"id": c["id"], "score": c["score"]} for c in chunks],
        ),
    }


def critic(state: AgentState, llm: LLM) -> dict:
    """Decide whether the retrieved chunks can answer the question."""
    top = state["chunks"][0]["score"]
    if top < SCORE_THRESHOLD:
        # WHY no LLM call: a score this low means nothing relevant came back.
        verdict, why = "insufficient", f"top score {top} is under {SCORE_THRESHOLD}"
    else:
        reply = parse_json(llm(
            CRITIQUE_PROMPT,
            f"Question: {state['question']}\n\nContext:\n{context(state['chunks'])}",
        ))
        # WHY "is True": anything unreadable counts as insufficient. Never guess.
        verdict = "sufficient" if reply.get("sufficient") is True else "insufficient"
        why = str(reply.get("reason", "the critique reply could not be read"))
    return {
        "critique": verdict,
        "trace": entry("critic", verdict, reason=why, top_score=top),
    }


def rewrite(state: AgentState, llm: LLM) -> dict:
    """Ask for a new search query in label vocabulary."""
    reply = llm(
        REWRITE_PROMPT,
        f"Question: {state['question']}\nPrevious query: {state['query']}",
    ).strip()
    # WHY first line only: models sometimes add an explanation underneath.
    query = reply.splitlines()[0].strip('" ') if reply else state["question"]
    return {
        "query": query,
        "trace": entry("rewrite", f"new query: {query}", previous=state["query"]),
    }


def write_answer(state: AgentState, llm: LLM) -> dict:
    """Answer from the chunks only, with citations."""
    reply = llm(
        ANSWER_PROMPT,
        f"Question: {state['question']}\n\nContext:\n{context(state['chunks'])}",
    ).strip()
    if not reply or reply.startswith("INSUFFICIENT"):
        # WHY a second check: the critique can be wrong. The writer gets one
        # last chance to say the context does not hold the answer.
        update = low_confidence("The retrieved label text does not contain the answer.")
    else:
        update = {"answer": reply, "confidence": "ok"}
    update["trace"] = entry(
        "write_answer", f"confidence = {update['confidence']}",
        sources=[c["id"] for c in state["chunks"]],
    )
    return update


def context(chunks: list[dict]) -> str:
    """Lay the chunks out so the model can cite each one by its id."""
    return "\n\n".join(
        f"[{c['id']}] ({c['metadata']['drug']}, {c['metadata']['section']}, "
        f"{c['metadata']['product_type']})\n{c['text']}"
        for c in chunks
    )


def safety_tool(state: AgentState) -> dict:
    """Run the graph safety check for the patient, drug and date."""
    args = (state["patient_id"], state["drug"], state["as_of"])
    return run_tool("safety_tool", tools.safety_check, args)


def risk_tool(state: AgentState) -> dict:
    """Look up the patient's readmission risk score."""
    return run_tool("risk_tool", tools.readmission_risk, (state["patient_id"],))


def run_tool(name: str, tool, args: tuple) -> dict:
    """Call a tool; turn its ValueError into a reason instead of a crash."""
    try:
        result = tool(*args)
    except ValueError as error:
        # WHY: unknown patient, unknown drug or a bad date must be refused,
        # not reported as "nothing found".
        return {
            "tool_result": None,
            "reason": f"The lookup failed: {error}.",
            "trace": entry(name, f"failed: {error}", args=list(args)),
        }
    return {
        "tool_result": result,
        "trace": entry(name, "ok", args=list(args), result=result),
    }


def tool_answer(state: AgentState) -> dict:
    """Turn a tool result into text without an LLM."""
    # WHY no LLM: the result is already exact and structured. A model could
    # only reword it, which adds a chance of error and nothing else.
    result = state["tool_result"]
    text = safety_text(result) if state["route"] == "safety" else risk_text(result)
    return {
        "answer": text,
        "confidence": "ok",
        "trace": entry("tool_answer", "formatted the tool result"),
    }


def safety_text(result: dict) -> str:
    """Describe a safety result, quoting the label sentence behind each flag."""
    lines = [
        f"Safety check for patient {result['patient_id']} and "
        f"{result['drug']}, as of {result['as_of']}:"
    ]
    for c in result["contraindications"]:
        lines.append(
            f"- Contraindication: {c['condition']} (active since {c['onset']}). "
            f"Label text: \"{c['source']}\""
        )
    for i in result["interactions"]:
        lines.append(
            f"- Interaction: the patient takes {i['drug']} (since {i['start']}). "
            f"Label text: \"{' | '.join(i['sources'])}\""
        )
    if not result["flagged"]:
        lines.append(NOT_SAFE_NOTE)
    return "\n".join(lines)


def risk_text(result: dict) -> str:
    """Describe a readmission-risk result and where it came from."""
    return (
        f"Patient {result['patient_id']} has a 30-day readmission risk score of "
        f"{result['score']} from their most recent admission "
        f"({result['admission_date']}; {result['admissions_on_file']} admissions "
        f"on file). The score comes from a model trained on synthetic data."
    )


def fallback(state: AgentState) -> dict:
    """Refuse, and say why."""
    reason = state.get("reason") or (
        f"The drug labels I searched did not clearly answer it after "
        f"{state.get('attempts', 0)} attempts (limit {MAX_ATTEMPTS}). {NARROW_HINT}"
    )
    return {**low_confidence(reason), "trace": entry("fallback", reason)}
