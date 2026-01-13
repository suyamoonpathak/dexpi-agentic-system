# DEXPI Agentic RAG System

A specialized Graph-RAG system for parsing, reasoning, and querying DEXPI P&ID (Process & Instrumentation Diagram) XML files. Built with LangGraph, LightRAG, and Llama 3.

## Features
- **Deep XML Parsing:** Extracts Equipment, Piping, Attributes, and Topology.
- **Graph-RAG Engine:** Combines Vector Search (Semantics) with Graph Logic (Connectivity).
- **Agentic Workflow:** - **Router:** Classifies queries (Lookup vs. Reasoning).
  - **Retriever:** Uses hybrid search with custom vector decoding.
  - **Generator:** Enforces strict engineering grounding to prevent hallucinations.

## Architecture
1. **Ingestion Layer:** - Parses XML -> Generates Semantic Sentences (Rich Text) -> Embeds into LightRAG.
2. **Retrieval Layer:**
   - Custom `retrieve_context` function bypasses generic limits.
   - Decodes Base64/Zlib vectors for precise Cosine Similarity.
3. **Agent Layer:**
   - LangGraph State Machine manages the flow: `Router -> Retriever -> Generator`.

## 📦 Installation

1. **Prerequisites:**
   - Python 3.12+
   - Ollama running locally or remotely (Llama 3, Nomic-Embed-Text)

2. **Setup:**
   ```bash
   poetry install
   cp .env.example .env  # Configure your OLLAMA_IP