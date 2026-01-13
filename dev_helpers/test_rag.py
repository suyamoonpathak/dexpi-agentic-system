import sys
import os
import logging
import time
import json
import asyncio

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ingestion.xml_parser import DexpiParser
from src.rag.engine import LightRAGEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RAG_Test")

async def main():
    xml_file = "data/raw/C01V01-HEX.EX03.xml"

    logger.info(f"Step 1: Parsing {xml_file}...")
    parser = DexpiParser()
    parsed_data = parser.parse_file(xml_file)
    
    logger.info("--- Data Quality Check ---")
    valid_items = [item for item in parsed_data['equipment'] if item['tag'] != "Unknown"]
    logger.info(f"Found {len(parsed_data['equipment'])} total items, {len(valid_items)} valid named items.")
    
    # Initialize LightRAG
    logger.info("Step 2: Initializing LightRAG Engine...")
    engine = LightRAGEngine()
    
    await engine.initialize()
    
    # Ingest Data
    logger.info("Step 3: Ingesting into Graph (This may take 30-60 seconds)...")
    start_time = time.time()
    
    await engine.ingest_dexpi_data(parsed_data)
    
    elapsed = time.time() - start_time
    logger.info(f"Ingestion complete in {elapsed:.2f} seconds.")

    # Test Query
    query = "What components are connected to piping segment 47124_1?"
    logger.info(f"Step 4: Running Query: '{query}'")
    
    response = await engine.query(query, mode="hybrid")
    
    print("\n" + "="*50)
    print(f"QUERY: {query}")
    print("-" * 50)
    print(f"ANSWER:\n{response}")
    print("="*50 + "\n")

if __name__ == "__main__":
    asyncio.run(main())