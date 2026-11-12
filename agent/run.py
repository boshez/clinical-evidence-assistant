"""Ask the agent one question.

Run with:  python -m agent.run "What is metformin indicated for?"
"""
import sys
from datetime import datetime

from agent.graph import build_graph
from agent.llm import LLM, chat
from agent.trace import save


def run_agent(question: str, llm: LLM = chat) -> dict:
    """Run one question through the graph and save its trace."""
    state = build_graph(llm).invoke({"question": question, "trace": []})
    # WHY a timestamp id: trace files sort in the order the questions were asked.
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    state["trace_file"] = str(save(run_id, state["trace"]))
    return state


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit('usage: python -m agent.run "your question"')
    result = run_agent(sys.argv[1])
    for step, record in enumerate(result["trace"], start=1):
        print(f"{step}. {record['node']}: {record['summary']}")
    print(f"\nconfidence: {result['confidence']}\n\n{result['answer']}")
    print(f"\ntrace saved to {result['trace_file']}")
