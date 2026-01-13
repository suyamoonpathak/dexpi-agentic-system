import time
from src.agents.state import AgentState
from src.rag.engine import LightRAGEngine
from src.ingestion.graph_builder import PIDGraphBuilder
from src.utils.monitor import monitor
import networkx as nx

rag_engine = LightRAGEngine()

graph_tool = PIDGraphBuilder() 
graph_tool.load_graph()

async def router_node(state: AgentState) -> AgentState:
    t0 = time.time()
    q = state['question'].lower()
    
    # Conditional Logic
    # If the user asks about connections, paths, or neighbors -> TOPOLOGY
    if any(x in q for x in ['connected', 'path', 'between', 'downstream', 'upstream', 'next to']):
        intent = "topology_agent"
    else:
        intent = "semantic_agent"
        
    monitor.log_step("Router", state['question'], intent, t0)
    return {"intent": intent, "steps": ["Router"]}

async def topology_node(state: AgentState) -> AgentState:
    """
    Agent responsible for Graph Theory queries using NetworkX.
    Now uses 'Smart Lookup' to handle complex IDs and Tags.
    """
    t0 = time.time()
    question = state['question']
    
    # 1. Ask the Graph to identify relevant entities in the text
    relevant_ids = graph_tool.find_relevant_nodes(question)
    
    context = ""
    
    if not relevant_ids:
        context = "Topological Analysis: No specific equipment IDs or Tags (like P-101) were found in your query to trace."
        
    elif len(relevant_ids) == 1:
        # Case A: "What is connected to X?"
        target_id = relevant_ids[0]
        neighbors = graph_tool.get_neighbors_by_id(target_id)
        context = "\n".join(neighbors)
        
    elif len(relevant_ids) >= 2:
        # Case B: "Path between X and Y"
        # We try to find path between the first two identified nodes
        start_id, end_id = relevant_ids[0], relevant_ids[1]
        
        try:
            path = nx.shortest_path(graph_tool.graph, start_id, end_id)
            path_desc = []
            for pid in path:
                p_data = graph_tool.graph.nodes[pid]
                path_desc.append(f"{p_data.get('tag', 'Unknown')} ({p_data.get('type')})")
            context = "Path Found:\n" + " -> ".join(path_desc)
        except Exception:
            context = f"No direct physical connection found between the identified items."

    monitor.log_step("Topology_Agent", question, context, t0)
    return {"context": f"Topological Analysis Result:\n{context}", "steps": ["Topology_Agent"]}

async def semantic_node(state: AgentState) -> AgentState:
    """
    Agent responsible for Conceptual/Attribute queries using LightRAG.
    """
    t0 = time.time()
    answer = await rag_engine.query_hybrid(state['question'])
    
    monitor.log_step("Semantic_Agent", state['question'], answer, t0)
    return {"final_answer": answer, "steps": ["Semantic_Agent"]}

async def synthesizer_node(state: AgentState) -> AgentState:
    """
    Only used if Topology Agent ran, to humanize the NetworkX output.
    """
    return {"final_answer": state['context'], "steps": ["Synthesizer"]}