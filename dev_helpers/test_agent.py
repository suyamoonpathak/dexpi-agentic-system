import asyncio
import logging
from src.agents.workflow import build_agent_graph

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

async def main():
    app = build_agent_graph()
    
    # Test Query 1: Specific Lookup
    q1 = "What is the ID of Nozzle_e36d2abc-bac7-438d-9dab-0d79054a53a6?"
    # What is Nozzle_e36d2abc-bac7-438d-9dab-0d79054a53a6 connected to?
    # What is the ID of Nozzle_e36d2abc-bac7-438d-9dab-0d79054a53a6?

    print(f"\n💬 Query 1: {q1}")
    print("-" * 50)
    
    inputs = {"question": q1, "steps": []}
    
    result = await app.ainvoke(inputs)
    
    print("-" * 50)
    print(f"Reasoning Trace: {result['steps']}")
    print(f"Intent Detected: {result['intent']}")
    print(f"Final Answer:\n{result['final_answer']}")
    print("=" * 50)

if __name__ == "__main__":
    asyncio.run(main())