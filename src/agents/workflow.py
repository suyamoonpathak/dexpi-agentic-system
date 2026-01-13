from langgraph.graph import StateGraph, END
from src.agents.state import AgentState
from src.agents.nodes import router_node, topology_node, semantic_node, synthesizer_node

def build_agent_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("router", router_node)
    workflow.add_node("topology_agent", topology_node)
    workflow.add_node("semantic_agent", semantic_node)
    workflow.add_node("synthesizer", synthesizer_node)

    workflow.set_entry_point("router")

    # CONDITIONAL ROUTING
    workflow.add_conditional_edges(
        "router",
        lambda x: x['intent'], 
        {
            "topology_agent": "topology_agent",
            "semantic_agent": "semantic_agent"
        }
    )

    # Topology needs synthesis, Semantic (LightRAG) is already synthesized
    workflow.add_edge("topology_agent", "synthesizer")
    workflow.add_edge("synthesizer", END)
    workflow.add_edge("semantic_agent", END)

    return workflow.compile()