from typing import Annotated, List, TypedDict, Any
import operator

class AgentState(TypedDict):
    """
    The shared memory state of the agent.
    """
    question: str                
    intent: str                  # 'specific_lookup' or 'graph_reasoning'
    context: str                
    final_answer: str            
    steps: List[str]             # decision trace logs
    graph_data: Any