import os
import json
import logging
import base64
import zlib
import numpy as np
from typing import List, Dict, Any
from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc
from config.settings import get_settings
from src.llm.ollama_client import ollama_llm_func, ollama_embedding_func

logger = logging.getLogger(__name__)
settings = get_settings()

class LightRAGEngine:
    def __init__(self):
        self.working_dir = settings.WORKING_DIR
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)

        embedding_func_wrapper = EmbeddingFunc(
            embedding_dim=768, 
            max_token_size=8192, 
            func=ollama_embedding_func 
        )

        self.rag = LightRAG(
            working_dir=self.working_dir,
            llm_model_func=ollama_llm_func,
            embedding_func=embedding_func_wrapper,
        )

    async def initialize(self):
        """Ensures storage is ready."""
        await self.rag.initialize_storages()

    # =========================================================================
    #  INGESTION: RICH TEXT SYNTHESIS
    # =========================================================================
    async def ingest_dexpi_data(self, parsed_data: Dict[str, Any]):
        """
        Converts parsed dictionary into rich, semantic sentences for embedding.
        Dynamically handles all attributes to prevent information loss.
        """
        logger.info("Starting rich text synthesis for ingestion...")
        filename = parsed_data['filename']
        texts_to_ingest = []
        prefix = "Context: Process Engineering P&ID Data. "

        # 1. Equipment & Attributes
        for item in parsed_data['equipment']:
            if item['tag'] == "Unnamed_Component" and not item['attributes']: 
                continue

            tag = item.get('tag', 'Unknown')
            etype = item.get('type', 'Unknown')
            
            # Enrichment: Add functional roles
            role = ""
            if "Valve" in etype: role = " It acts as a flow control/isolation device."
            elif "Pump" in etype: role = " It provides pressure to move fluids."
            elif "Tank" in etype: role = " It stores fluid inventory."
            
            base_desc = f"{prefix}In '{filename}', entity '{tag}' (ID: {item['id']}) is a '{etype}'.{role}"
            
            # Dynamic Attribute Loop
            attr_text = []
            if item.get('attributes'):
                for k, v in item['attributes'].items():
                    # Clean technical keys for better embedding
                    clean_k = k.replace("AssignmentClass", "").replace("Specialization", "")
                    attr_text.append(f"Its {clean_k} is '{v}'.")
            
            full_text = f"{base_desc} {' '.join(attr_text)}"
            texts_to_ingest.append(full_text)

        # 2. Connections
        id_map = {i['id']: {"tag": i.get('tag', 'Unknown'), "type": i.get('type', 'Unknown')} 
                  for i in parsed_data['equipment']}

        for conn in parsed_data['connections']:
            src, tgt = conn['source'], conn['target']
            if src in id_map and tgt in id_map:
                s_info = id_map[src]
                t_info = id_map[tgt]
                rel_text = (f"{prefix}Connection: The {s_info['type']} '{s_info['tag']}' (ID: {src}) "
                            f"is connected to the {t_info['type']} '{t_info['tag']}' (ID: {tgt}).")
                texts_to_ingest.append(rel_text)

        logger.info(f"Generated {len(texts_to_ingest)} synthetic sentences.")
        if texts_to_ingest:
            await self.rag.ainsert(texts_to_ingest)
        else:
            logger.warning("No text generated for ingestion.")

    # =========================================================================
    #  RETRIEVAL: CUSTOM VECTOR LOGIC
    # =========================================================================
    def _decode_vector(self, vec_raw):
        """Decodes LightRAG vectors: Base64 -> Zlib -> Float16 -> Float32"""
        if isinstance(vec_raw, list): return np.array(vec_raw, dtype=np.float32)
        if isinstance(vec_raw, str):
            try:
                vec_bytes = base64.b64decode(vec_raw)
                try: vec_bytes = zlib.decompress(vec_bytes)
                except: pass
                return np.frombuffer(vec_bytes, dtype=np.float16).astype(np.float32)
            except: pass
        return np.zeros(1)

    def _cosine_similarity(self, v1, v2):
        if v1.shape != v2.shape: return 0.0
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        return np.dot(v1, v2) / norm if norm != 0 else 0.0

    def _recursive_find_chunks(self, data, chunks_list):
        """Recursively scans JSON for objects with 'vector' and 'content'"""
        if isinstance(data, dict):
            if "vector" in data and "content" in data:
                chunks_list.append(data)
            for v in data.values():
                self._recursive_find_chunks(v, chunks_list)
        elif isinstance(data, list):
            for item in data:
                self._recursive_find_chunks(item, chunks_list)

    async def retrieve_context(self, query: str, top_k: int = 5) -> str:
        """
        Manually retrieves context from LightRAG storage to bypass internal query limitations.
        """
        logger.info(f"Retrieving context for: '{query}'")
        
        # 1. Load Data
        vdb_path = os.path.join(self.working_dir, "vdb_chunks.json")
        if not os.path.exists(vdb_path):
            return "Error: Database not found. Please ingest data first."
            
        try:
            with open(vdb_path, "r") as f:
                raw_json = json.load(f)
        except Exception as e:
            return f"Error loading database: {e}"
            
        chunks = []
        self._recursive_find_chunks(raw_json, chunks)
        
        if not chunks:
            return "No data chunks found in database."

        # 2. Embed Query
        query_vecs = await ollama_embedding_func([query])
        query_vec = np.array(query_vecs[0], dtype=np.float32)

        # 3. Rank
        scores = []
        for ch in chunks:
            vec = self._decode_vector(ch["vector"])
            score = self._cosine_similarity(query_vec, vec)
            scores.append({"score": score, "content": ch.get("content", "")})

        scores.sort(key=lambda x: x['score'], reverse=True)
        
        # 4. Format
        top_chunks = scores[:top_k]
        context_str = "\n\n".join([
            f"[Match (Score: {c['score']:.4f})]: {c['content']}" 
            for c in top_chunks
        ])
        
        logger.info(f"Retrieved {len(top_chunks)} chunks. Best score: {top_chunks[0]['score']:.4f}")
        return context_str