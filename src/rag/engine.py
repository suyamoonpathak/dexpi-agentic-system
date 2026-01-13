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
    #  INGESTION
    # =========================================================================
    async def ingest_dexpi_data(self, parsed_data: Dict[str, Any]):
        """Ingest data into LightRAG (remains unchanged)"""
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
            
            role = ""
            if "Valve" in etype: role = " It acts as a flow control/isolation device."
            elif "Pump" in etype: role = " It provides pressure to move fluids."
            elif "Tank" in etype: role = " It stores fluid inventory."
            
            base_desc = f"{prefix}In '{filename}', entity '{tag}' (ID: {item['id']}) is a '{etype}'.{role}"
            
            attr_text = []
            if item.get('attributes'):
                for k, v in item['attributes'].items():
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
    #  MANUAL RETRIEVAL HELPERS 
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

    def _recursive_find_chunks(self, data, chunks_list, key_field="content"):
        """Recursively scans JSON for objects with 'vector' and content"""
        if isinstance(data, dict):
            if "vector" in data and key_field in data:
                chunks_list.append(data)
            for v in data.values():
                self._recursive_find_chunks(v, chunks_list, key_field)
        elif isinstance(data, list):
            for item in data:
                self._recursive_find_chunks(item, chunks_list, key_field)

    async def _manual_search(self, filename: str, query_vec: np.ndarray, top_k: int = 5, key_field="content") -> List[str]:
        """Generic searcher for any LightRAG JSON storage."""
        path = os.path.join(self.working_dir, filename)
        if not os.path.exists(path):
            logger.warning(f"Storage file {filename} not found.")
            return []
            
        try:
            with open(path, "r") as f:
                raw_json = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load {filename}: {e}")
            return []
            
        items = []
        self._recursive_find_chunks(raw_json, items, key_field)
        
        scores = []
        for item in items:
            vec = self._decode_vector(item["vector"])
            score = self._cosine_similarity(query_vec, vec)
            # Store content and score
            content = item.get(key_field, "")
            scores.append({"score": score, "content": content})

        scores.sort(key=lambda x: x['score'], reverse=True)
        return [x['content'] for x in scores[:top_k]]

    # =========================================================================
    #  HYBRID QUERY LOGIC 
    # =========================================================================
    async def query_hybrid(self, query: str) -> str:
        """
        Manually implements Hybrid RAG to avoid async library bugs.
        1. Embeds Query.
        2. Searches Entities 
        3. Searches Text Chunks
        4. Synthesizes Answer.
        """
        logger.info(f"Executing Robust Manual Hybrid Query for: {query}")
        
        # 1. Embed Query
        try:
            query_vecs = await ollama_embedding_func([query])
            query_vec = np.array(query_vecs[0], dtype=np.float32)
        except Exception as e:
            return f"Error embedding query: {e}"

        # 2. Retrieve from Entities

        entities = await self._manual_search("vdb_entities.json", query_vec, top_k=5, key_field="content")
        
        # 3. Retrieve from Chunks (Raw Context)
        chunks = await self._manual_search("vdb_chunks.json", query_vec, top_k=5, key_field="content")

        # 4. Retrieve from Relationships (optional but good for 'hybrid')
        relations = await self._manual_search("vdb_relationships.json", query_vec, top_k=3, key_field="content")

        if not entities and not chunks:
            return "No relevant data found in the knowledge base."

        # 5. Synthesize Context
        context_parts = []
        if entities:
            context_parts.append("--- IDENTIFIED ENTITIES ---")
            context_parts.extend(entities)
        
        if chunks:
            context_parts.append("\n--- RELEVANT EXCERPTS ---")
            context_parts.extend(chunks)
        
        if relations:
            context_parts.append("\n--- RELEVANT EXCERPTS ---")
            context_parts.extend(relations)

        full_context = "\n".join(context_parts)
        
        # 6. Generate Answer
        prompt = f"""
        You are a Senior Process Engineer. Answer based ONLY on the context below.
        
        CONTEXT:
        {full_context}
        
        QUESTION: 
        {query}
        
        ANSWER (Be precise, list IDs if found):
        """
        
        answer = await ollama_llm_func(prompt)
        return answer

    async def retrieve_context(self, query: str, top_k: int = 5) -> str:
        query_vecs = await ollama_embedding_func([query])
        query_vec = np.array(query_vecs[0], dtype=np.float32)
        chunks = await self._manual_search("vdb_chunks.json", query_vec, top_k=top_k)
        return "\n\n".join(chunks)