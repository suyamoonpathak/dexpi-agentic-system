import asyncio
import logging
import argparse
import sys
import os

# Ensure src is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.agents.workflow import build_agent_graph
from src.ingestion.xml_parser import DexpiParser
from src.ingestion.graph_builder import PIDGraphBuilder
from src.rag.engine import LightRAGEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("DEXPI_System")

async def run_ingestion(file_path: str):
    """
    Master Ingestion Pipeline:
    1. Parse XML -> Raw Dict
    2. Build NetworkX Graph -> Save .gpickle -> Draw .svg
    3. Build LightRAG Index -> Save Vector DB
    """
    logger.info(f"🚀 Starting System Ingestion for: {file_path}")
    
    # 1. Parsing
    parser = DexpiParser()
    try:
        data = parser.parse_file(file_path)
    except Exception as e:
        logger.error(f"Parsing failed: {e}")
        return

    # 2. Topological Ingestion (NetworkX)
    logger.info("--- Phase 1: Topological Processing ---")
    gb = PIDGraphBuilder()
    gb.build_graph(data)
    gb.save_graph()  # <--- CRITICAL: Save to disk
    gb.generate_visualization(output_path="data/processed/topology.svg")

    # 3. Semantic Ingestion (LightRAG)
    logger.info("--- Phase 2: Semantic Processing ---")
    engine = LightRAGEngine()
    await engine.initialize()
    await engine.ingest_dexpi_data(data)
    
    logger.info("Ingestion Complete. System ready for queries.")

async def run_agent(query: str):
    """
    Query Pipeline:
    1. Load Graphs (NetworkX + LightRAG)
    2. Start Agent Workflow
    """
    logger.info(f"Agent started with query: '{query}'")
    
    # The agents inside build_agent_graph will load the data they need
    app = build_agent_graph()
    
    inputs = {"question": query, "steps": []}
    
    try:
        result = await app.ainvoke(inputs)
        
        print("\n" + "="*60)
        print(f"🚦 Final Intent: {result['intent'].upper()}")
        print("-" * 60)
        if result.get('final_answer'):
            print(f"Answer:\n{result['final_answer']}")
        else:
            print(f"Context:\n{result.get('context', 'No answer generated')}")
        print("="*60 + "\n")
        
    except Exception as e:
        logger.error(f"Agent execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DEXPI Agentic System")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest a DEXPI XML file")
    ingest_parser.add_argument("file", help="Path to .xml file")

    query_parser = subparsers.add_parser("query", help="Ask a question")
    query_parser.add_argument("text", help="The question string")

    args = parser.parse_args()

    if args.command == "ingest":
        asyncio.run(run_ingestion(args.file))
    elif args.command == "query":
        asyncio.run(run_agent(args.text))