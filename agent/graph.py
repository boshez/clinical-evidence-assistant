"""Wire the nodes into a LangGraph graph. The three decisions live here."""
from functools import partial

from langgraph.graph import END, START, StateGraph

from agent import nodes
from agent.llm import LLM, chat
from agent.state import MAX_ATTEMPTS, AgentState

# NOTE: node names differ from state field names (router/route, critic/critique)
# because LangGraph does not allow a node and a state field to share a name.


def after_router(state: AgentState) -> str:
    """Decision 1: send the question to the tool the router picked."""
    return {
        "guideline": "retrieve",
        "safety": "safety_tool",
        "risk": "risk_tool",
        "refuse": "fallback",
    }[state["route"]]


def after_critic(state: AgentState) -> str:
    """Decision 2: answer, try again, or give up."""
    if state["critique"] == "sufficient":
        return "write_answer"
    # WHY count in state: the limit holds however the loop is entered, and
    # the trace shows exactly how many attempts were made.
    if state["attempts"] < MAX_ATTEMPTS:
        return "rewrite"
    return "fallback"


def after_tool(state: AgentState) -> str:
    """Decision 3: a tool that failed must refuse, not answer."""
    return "tool_answer" if state.get("tool_result") else "fallback"


def build_graph(llm: LLM = chat):
    """Build and compile the agent. Tests pass their own llm."""
    graph = StateGraph(AgentState)
    # WHY partial: LangGraph calls a node with the state only, so the llm is
    # fixed in advance for the nodes that need one.
    graph.add_node("router", partial(nodes.router, llm=llm))
    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("critic", partial(nodes.critic, llm=llm))
    graph.add_node("rewrite", partial(nodes.rewrite, llm=llm))
    graph.add_node("write_answer", partial(nodes.write_answer, llm=llm))
    graph.add_node("safety_tool", nodes.safety_tool)
    graph.add_node("risk_tool", nodes.risk_tool)
    graph.add_node("tool_answer", nodes.tool_answer)
    graph.add_node("fallback", nodes.fallback)

    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router", after_router,
        ["retrieve", "safety_tool", "risk_tool", "fallback"],
    )
    graph.add_edge("retrieve", "critic")
    graph.add_conditional_edges(
        "critic", after_critic, ["write_answer", "rewrite", "fallback"]
    )
    graph.add_edge("rewrite", "retrieve")  # the retry loop
    for tool in ("safety_tool", "risk_tool"):
        graph.add_conditional_edges(tool, after_tool, ["tool_answer", "fallback"])
    for last in ("write_answer", "tool_answer", "fallback"):
        graph.add_edge(last, END)
    return graph.compile()
