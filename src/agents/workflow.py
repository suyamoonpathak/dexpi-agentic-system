from langgraph.graph import StateGraph, END
from src.agents.state import AgentState
from src.agents.nodes import router_node, retriever_node, generator_node

def build_agent_graph():
    """
    Constructs the LangGraph state machine.
    """
    workflow = StateGraph(AgentState)

    # 1. Add Nodes
    workflow.add_node("router", router_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("generator", generator_node)

    # 2. Entry Point
    workflow.set_entry_point("router")

    # 3. Edges
    workflow.add_edge("router", "retriever")
    
    # From Retriever -> Generator
    workflow.add_edge("retriever", "generator")
    
    # From Generator -> End
    workflow.add_edge("generator", END)

    # 4. Compile
    return workflow.compile()