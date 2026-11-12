"""Tests for the agent's wiring: routes, the retry limit and the refusals.

Run with:  python -m agent.test_agent

The LLM is replaced by a scripted stand-in, so these tests check the GRAPH
(which node runs next, and when it stops), not the model's judgement. That
makes them repeatable and free. Retrieval, Neo4j and the risk CSV are real.
Model quality is measured separately in the evaluation step.
"""
import json
from pathlib import Path

from agent import nodes
from agent.run import run_agent
from agent.state import MAX_ATTEMPTS


def scripted(route: dict | str, sufficient: bool = True,
             answer: str = "Stub answer [chunk-id].") -> tuple:
    """Build a stand-in LLM and the list that records which prompts it saw."""
    seen: list[str] = []

    def llm(system: str, user: str) -> str:
        if system is nodes.ROUTE_PROMPT:
            seen.append("route")
            return route if isinstance(route, str) else json.dumps(route)
        if system is nodes.CRITIQUE_PROMPT:
            seen.append("critique")
            return json.dumps({"sufficient": sufficient, "reason": "scripted"})
        if system is nodes.REWRITE_PROMPT:
            seen.append("rewrite")
            return f"rewritten query {seen.count('rewrite')}"
        seen.append("answer")
        return answer

    return llm, seen


def path(state: dict) -> list[str]:
    """The node names in the order they ran."""
    return [record["node"] for record in state["trace"]]


def test_answerable_question_answers_first_time() -> None:
    """1. Good retrieval + sufficient critique: one attempt, then answer."""
    llm, _ = scripted({"route": "guideline"})
    state = run_agent("What is metformin indicated for?", llm)
    assert path(state) == ["router", "retrieve", "critic", "write_answer"]
    assert state["attempts"] == 1 and state["confidence"] == "ok"


def test_retries_stop_at_the_limit() -> None:
    """2. Critique never satisfied: exactly 3 retrievals, 2 rewrites, refuse."""
    llm, seen = scripted({"route": "guideline"}, sufficient=False)
    state = run_agent("What is metformin indicated for?", llm)
    assert state["attempts"] == MAX_ATTEMPTS == 3
    assert path(state).count("retrieve") == 3 and seen.count("rewrite") == 2
    assert path(state)[-1] == "fallback" and state["confidence"] == "low"
    assert "answer" not in seen  # the writer was never asked


def test_low_score_skips_the_critique_model() -> None:
    """3. Off-topic text scores under the threshold: refuse, no critique call."""
    llm, seen = scripted({"route": "guideline"})
    state = run_agent("zzz qqq xyzzy plugh", llm)
    assert "critique" not in seen and "answer" not in seen
    assert state["confidence"] == "low"


def test_writer_can_still_refuse() -> None:
    """4. Critique says yes but the writer says INSUFFICIENT: low confidence."""
    llm, _ = scripted({"route": "guideline"}, answer="INSUFFICIENT")
    state = run_agent("What is metformin indicated for?", llm)
    assert path(state)[-1] == "write_answer" and state["confidence"] == "low"


def test_unreadable_route_falls_back_to_retrieval() -> None:
    """5. Router reply is not JSON: use the guideline route, which can refuse."""
    llm, _ = scripted("sorry, I am not sure")
    state = run_agent("What is metformin indicated for?", llm)
    assert state["route"] == "guideline"


def test_safety_flags_active_condition() -> None:
    """6. Safety route, active condition: flagged, with the label sentence."""
    llm, seen = scripted({"route": "safety", "patient_id": "TEST_ACTIVE",
                          "drug": "Spironolactone", "as_of": "2026-06-01"})
    state = run_agent("Can TEST_ACTIVE take spironolactone on 2026-06-01?", llm)
    assert path(state) == ["router", "safety_tool", "tool_answer"]
    assert state["tool_result"]["flagged"] and "hyperkalemia" in state["answer"]
    assert seen == ["route"]  # no model wrote the safety answer


def test_safety_resolved_condition_never_says_safe() -> None:
    """7. Resolved condition: not flagged, and the answer does not claim safety."""
    llm, _ = scripted({"route": "safety", "patient_id": "TEST_RESOLVED",
                       "drug": "spironolactone", "as_of": "2026-06-01"})
    state = run_agent("Can TEST_RESOLVED take spironolactone?", llm)
    assert not state["tool_result"]["flagged"]
    assert "not a statement that the drug is safe" in state["answer"]


def test_unknown_patient_is_refused() -> None:
    """8. Unknown patient: refuse. It must not read as 'nothing found'."""
    llm, _ = scripted({"route": "safety", "patient_id": "NOBODY",
                       "drug": "warfarin", "as_of": "2026-06-01"})
    state = run_agent("Can NOBODY take warfarin?", llm)
    assert path(state)[-1] == "fallback" and "unknown patient" in state["answer"]


def test_safety_without_a_drug_is_refused() -> None:
    """9. Safety route with no drug named: refuse before calling the tool."""
    llm, _ = scripted({"route": "safety", "patient_id": "TEST_ACTIVE"})
    state = run_agent("Is TEST_ACTIVE safe?", llm)
    assert path(state) == ["router", "fallback"]


def test_risk_uses_latest_admission() -> None:
    """10. Risk route: the score comes from the most recent admission."""
    llm, _ = scripted({"route": "risk", "patient_id": "P00017"})
    state = run_agent("What is P00017's readmission risk?", llm)
    result = state["tool_result"]
    assert result["admissions_on_file"] > 1
    assert result["admission_date"] > "2025-03-12"  # later than the first two rows
    assert str(result["score"]) in state["answer"]


def test_loosely_typed_patient_id_is_tidied() -> None:
    """13. '00017' and 'p17' both mean P00017; TEST_ ids are left alone."""
    for typed in ("00017", "p17", "P00017"):
        llm, _ = scripted({"route": "risk", "patient_id": typed})
        state = run_agent("readmission risk?", llm)
        assert state["patient_id"] == "P00017" and state["confidence"] == "ok"
    assert nodes.clean_patient_id("test_active") == "TEST_ACTIVE"


def test_off_topic_route_is_refused() -> None:
    """11. Router says refuse: straight to fallback, nothing retrieved."""
    llm, _ = scripted({"route": "refuse"})
    state = run_agent("What is the capital of France?", llm)
    assert path(state) == ["router", "fallback"] and state["confidence"] == "low"


def test_trace_file_matches_the_run() -> None:
    """12. The saved trace has one JSON line per node that ran."""
    llm, _ = scripted({"route": "refuse"})
    state = run_agent("What is the capital of France?", llm)
    lines = Path(state["trace_file"]).read_text().splitlines()
    assert [json.loads(line)["node"] for line in lines] == path(state)


if __name__ == "__main__":
    tests = [value for name, value in list(globals().items())
             if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS  {test.__doc__.splitlines()[0]}")
    print(f"\n{len(tests)}/{len(tests)} passed")
