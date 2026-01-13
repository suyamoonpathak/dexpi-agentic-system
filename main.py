import asyncio
import logging
import argparse
import sys
from src.agents.workflow import build_agent_graph
from src.ingestion.xml_parser import DexpiParser
from src.rag.engine import LightRAGEngine

# Configure professional logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("DEXPI_Agent")

async def run_ingestion(file_path: str):
    """Parses XML and ingests into Knowledge Base."""
    logger.info(f"🚀 Starting Ingestion for: {file_path}")
    
    # 1. Parse
    parser = DexpiParser()
    try:
        data = parser.parse_file(file_path)
    except Exception as e:
        logger.error(f"Parsing failed: {e}")
        return

    # 2. Ingest
    engine = LightRAGEngine()
    await engine.initialize()
    await engine.ingest_dexpi_data(data)
    logger.info("✅ Ingestion Complete. Knowledge Base updated.")

async def run_agent(query: str):
    """Runs the Agentic Workflow."""
    logger.info(f"🤖 Agent started with query: '{query}'")
    
    app = build_agent_graph()
    inputs = {"question": query, "steps": []}
    
    try:
        result = await app.ainvoke(inputs)
        
        print("\n" + "="*60)
        print(f"🚦 Strategy: {result['intent'].upper()}")
        print("-" * 60)
        print(f"📝 Answer:\n{result['final_answer']}")
        print("="*60 + "\n")
        
    except Exception as e:
        logger.error(f"Agent execution failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DEXPI Agentic RAG System")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: ingest
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a DEXPI XML file")
    ingest_parser.add_argument("file", help="Path to .xml file")

    # Subcommand: query
    query_parser = subparsers.add_parser("query", help="Ask a question to the agent")
    query_parser.add_argument("text", help="The question string")

    args = parser.parse_args()

    if args.command == "ingest":
        asyncio.run(run_ingestion(args.file))
    elif args.command == "query":
        asyncio.run(run_agent(args.text))