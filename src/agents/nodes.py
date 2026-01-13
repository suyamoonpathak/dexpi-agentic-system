import logging
from src.agents.state import AgentState
from src.rag.engine import LightRAGEngine
from src.llm.ollama_client import ollama_llm_func

logger = logging.getLogger(__name__)
rag_engine = LightRAGEngine()

async def router_node(state: AgentState) -> AgentState:
    logger.info(f"[Router] Analyzing query: {state['question']}")
    
    q_lower = state['question'].lower()
    keywords = ['connected', 'flow', 'path', 'isolation', 'between', 'trace']
    
    intent = "graph_reasoning" if any(k in q_lower for k in keywords) else "specific_lookup"

    logger.info(f"[Router] Decision: {intent}")
    return {"intent": intent, "steps": ["Router"]}

async def retriever_node(state: AgentState) -> AgentState:
    question = state["question"]
    
    await rag_engine.initialize()
    
    context = await rag_engine.retrieve_context(question, top_k=7)

    logger.info(f"[Retriever] Retrieved context for question: \n\n {context} \n\n")

    return {"context": context, "steps": ["Custom_Retriever"]}

async def generator_node(state: AgentState) -> AgentState:
    logger.info("[Generator] Generating answer...")
    
    prompt = f"""
    You are a Senior Process Engineer analyzing a DEXPI P&ID database.
    
    CONTEXT (Extracted from Plant Database):
    {state['context']}
    
    USER QUESTION: 
    {state['question']}
    
    INSTRUCTIONS:
    1. Answer strictly based on the Context provided above.
    2. If the user asks about a specific ID or Tag, look for it exactly in the text.
    3. If attributes (like Pressure, Temperature, Material) are requested, extract them directly.
    4. If the information is not in the context, explicitly state "Information not found in the parsed data."
    5. Also mention the source of your context like the File name and other details if present to make the system more observable.
    
    ANSWER:
    """
    
    answer = await ollama_llm_func(prompt)
    
    return {"final_answer": answer, "steps": ["Generator"]}