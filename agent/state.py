"""The agent's state: the one dictionary every node reads and updates."""
import operator
from typing import Annotated, Literal, TypedDict

# WHY constants here: the conditional edges and the tests both need them.
MAX_ATTEMPTS = 3  # 1 initial retrieval + 2 rewrites
# Measured by eval/retrieval_eval.py: the 32 answerable gold questions score
# 0.535 or more and off-topic questions 0.163 or less, so 0.35 is the midpoint.
# It only catches off-topic questions. In-domain unanswerable ones score 0.54
# to 0.67, the same as real ones, so the critique step has to catch those.
SCORE_THRESHOLD = 0.35

Route = Literal["guideline", "safety", "risk", "refuse"]


class AgentState(TypedDict, total=False):
    """Everything the agent knows during one question."""

    question: str            # the user's words; never overwritten
    route: Route             # which tool the router picked
    patient_id: str | None   # for the safety and risk tools
    drug: str | None         # for the safety tool
    as_of: str | None        # 'YYYY-MM-DD' for the safety tool
    query: str               # current search query; replaced on rewrite
    attempts: int            # retrievals done so far (max MAX_ATTEMPTS)
    chunks: list[dict]       # latest retrieved chunks with scores
    critique: Literal["sufficient", "insufficient"]
    tool_result: dict | None  # output of the safety or risk tool
    reason: str              # why the agent fell back, if it did
    answer: str              # final text shown to the user
    confidence: Literal["ok", "low"]
    # WHY Annotated with operator.add: a node returns only its NEW trace
    # entries, and LangGraph appends them instead of replacing the list.
    trace: Annotated[list[dict], operator.add]
