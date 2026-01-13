# DEXPI Agentic System 🏭

A **RAG System** designed to query complex Process Engineering (P&ID) data. This system ingests DEXPI XML files and uses a multi-agent architecture to answer both topological (pathfinding) and semantic (conceptual) queries.

## 🌟 Key Features

* **Hybrid Architecture**: Combines **NetworkX** for precise physical tracing and **LightRAG** for semantic understanding.
* **Agentic Orchestration**: Uses **LangGraph** to conditionally route queries to specialized agents (Topology Agent vs. Semantic Agent).
* **Smart Graph Ingestion**: Automatically parses DEXPI XMLs, handling implicit piping segments and generating a clean directed graph.
* **Visual Observability**: Generates SVG visualizations of the P&ID topology upon ingestion.
* **Local Privacy**: Powered entirely by local LLMs via **Ollama** (Llama 3, Nomic Embed).


## 🚀 Prerequisites

1. **Python 3.10+**
2. **Ollama**: Installed and running locally.
* Pull the required models:
```bash
ollama pull llama3:8b
ollama pull nomic-embed-text

```





## 🛠️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/suyamoonpathak/dexpi-agentic-system
cd dexpi-agentic-system

```


2. **Install Dependencies from the poetry.lock file**
```bash
poetry install
```


3. **Environment Setup**
Create a `.env` file in the root directory:
```ini
OLLAMA_SERVER_IP=localhost
OLLAMA_PORT=11434
OLLAMA_MODEL=llama3:8b
OLLAMA_EMBED_MODEL=nomic-embed-text
WORKING_DIR=data/knowledge_base

```



## 📖 Usage

### 1. Ingestion Phase

This step parses the XML, builds the topological graph, and creates the vector embeddings.

```bash
python main.py ingest "data/raw/C01V01-HEX.EX03.xml"

```

* **Output:**
* `data/knowledge_base/system_topology.gpickle`: Saved graph object.
* `data/knowledge_base/vdb_*.json`: Vector stores.
* `data/processed/topology.svg`: Visual representation of the plant.



### 2. Query Phase

Ask questions to your agents. The system will automatically decide which strategy to use. Some queries are written in the queries.txt file with their correct answers.

**Topology Query (Pathfinding):**

```bash
python main.py query "Trace the flow path starting from BlindFlange SP3D14F870DEB44B129D3E37D162D6A661 to Nozzle SP7EB92322FAC14299AB96093CA149E484."
```

**Semantic Query (Attributes):**

```bash
python main.py query "What is the LowerLimitDesignPressure of the Tank with Tag T-4750?"

```

## 🧠 Technical Deep Dive

### The Dual-Graph Strategy

P&ID data requires two types of reasoning:

1. **Exact Match (Topology):** "Is Pump A connected to Tank B?" Vector databases fail at this because they look for semantic similarity, not physical connectivity. We solve this with **NetworkX**.
2. **Fuzzy Match (Semantics):** "List all safety valves." Graph databases struggle here if the schema isn't perfect. We solve this with **LightRAG** and synthetic sentence generation.

### Graph & Vector Visualization

Upon ingestion, the system uses `matplotlib` to render the NetworkX graph into an SVG file located in `data/processed/`. This provides immediate visual verification of the parsed data structure.

### Observability

The system includes a singleton `SystemMonitor` that tracks:

* Agent routing decisions.
* Step latency.
* Token usage (via Ollama logs).
Logs are printed to the console in real-time.

## 📂 Project Structure

```
.
├── config/             # Pydantic settings & Env vars
├── data/               # Raw XMLs and Processed KBs
├── src/
│   ├── agents/         # LangGraph Nodes & Workflow
│   ├── ingestion/      # XML Parser & NetworkX Builder
│   ├── llm/            # Async Ollama Client
│   └── rag/            # LightRAG Engine & Prompts
├── main.py             # CLI Entry Point
└── README.md           # Documentation

```
