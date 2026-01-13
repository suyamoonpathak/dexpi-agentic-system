import os
import json
import logging
import asyncio
import base64
import zlib
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from src.ingestion.xml_parser import DexpiParser
from src.rag.engine import LightRAGEngine
from src.llm.ollama_client import ollama_llm_func, ollama_embedding_func

# Setup
DEBUG_DIR = "debug_outputs"
KB_DIR = "data/knowledge_base"
if not os.path.exists(DEBUG_DIR):
    os.makedirs(DEBUG_DIR)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DebugPipeline")

# --- CUSTOM RETRIEVAL FUNCTIONS ---

def decode_vector(vec_raw):
    """
    Decodes LightRAG vectors.
    Correct format identified: Base64 -> Zlib -> Float16
    """
    if isinstance(vec_raw, list):
        return np.array(vec_raw, dtype=np.float32)
    
    if isinstance(vec_raw, str):
        try:
            # 1. Base64
            vec_bytes = base64.b64decode(vec_raw)
            # 2. Zlib
            try:
                vec_bytes = zlib.decompress(vec_bytes)
            except:
                pass
            
            # 3. Float16 -> Float32
            return np.frombuffer(vec_bytes, dtype=np.float16).astype(np.float32)
        except Exception as e:
            return np.zeros(1)
            
    return np.zeros(1)

def recursive_find_chunks(data, chunks_list):
    if isinstance(data, dict):
        if "vector" in data and "content" in data: 
            # Store ID if available for KV lookup
            if "id" not in data:
                 # Attempt to find key if this dict is a value in a larger dict
                 pass 
            chunks_list.append(data)
        for k, v in data.items():
            # If v is the chunk dict, k might be the ID. 
            # We can inject it if needed, but 'content' is usually inside 'data'.
            recursive_find_chunks(v, chunks_list)
    elif isinstance(data, list):
        for item in data:
            recursive_find_chunks(item, chunks_list)

def cosine_similarity(v1, v2):
    if v1.shape != v2.shape: return -1.0
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return 0.0 if (norm_v1 == 0 or norm_v2 == 0) else np.dot(v1, v2) / (norm_v1 * norm_v2)

async def custom_retriever(query: str, top_k: int = 5):
    """
    Manually retrieves relevant context from LightRAG storage 
    """
    print(f"Custom Retrieval for: '{query}'")
    
    # 1. Load Data
    vdb_path = os.path.join(KB_DIR, "vdb_chunks.json")
    if not os.path.exists(vdb_path):
        print("DB not found.")
        return ""
        
    with open(vdb_path, "r") as f:
        raw_json = json.load(f)
        
    chunks = []
    recursive_find_chunks(raw_json, chunks)
    print(f"Scanning {len(chunks)} vectors...")

    # 2. Embed Query
    query_vec = await ollama_embedding_func([query])
    query_vec = np.array(query_vec[0], dtype=np.float32)

    # 3. Rank
    scores = []
    for ch in chunks:
        vec = decode_vector(ch["vector"])
        score = cosine_similarity(query_vec, vec)
        
        # LightRAG sometimes stores content directly in vdb_chunks, 

        # Based on logs, 'content' key seems present in the chunk objects found.
        content = ch.get("content", "")
        
        scores.append({"score": score, "content": content})

    scores.sort(key=lambda x: x['score'], reverse=True)
    
    # 4. Format Output
    top_chunks = scores[:top_k]
    context_str = "\n\n".join([f"[Context #{i+1} (Score: {c['score']:.4f})]: {c['content']}" for i, c in enumerate(top_chunks)])
    
    print(f"Retrieved {len(top_chunks)} chunks. Top Score: {top_chunks[0]['score']:.4f}")
    return context_str


async def main():
    print("STARTING FULL DEBUG PIPELINE (CUSTOM RETRIEVAL)")
    print(f"All outputs will be saved to: {os.path.abspath(DEBUG_DIR)}\n")

    # =========================================================================
    # STEP 1: PARSING (XML -> Python Dict)
    # =========================================================================
    print("--- STEP 1: PARSING XML ---")
    xml_file = "data/raw/C01V01-HEX.EX03.xml"
    parser = DexpiParser()
    
    if not os.path.exists(xml_file):
        print(f"Error: File {xml_file} not found.")
        return

    parsed_data = parser.parse_file(xml_file)
    
    with open(f"{DEBUG_DIR}/1_parsed_data.json", "w") as f:
        json.dump(parsed_data, f, indent=2, default=str)
    
    print(f"Saved parsed data to '1_parsed_data.json'")


    # =========================================================================
    # STEP 2: SYNTHESIS (Python Dict -> English Sentences)
    # =========================================================================
    print("\n--- STEP 2: GENERATING RICH SYNTHETIC SENTENCES ---")
    
    filename = parsed_data['filename']
    texts_to_ingest = []
    prefix = "Context: Process Engineering P&ID Data. "

    # --- 2A. Equipment Sentences ---
    for item in parsed_data['equipment']:
        if item['tag'] == "Unnamed_Component" and not item['attributes']: 
            continue

        tag_clean = item['tag'] if item['tag'] else "Unknown_Tag"
        type_clean = item['type'] if item['type'] else "Unknown_Type"
        
        role_desc = ""
        if "Valve" in type_clean:
            role_desc = " It acts as a flow control or isolation device."
        elif "Pump" in type_clean:
            role_desc = " It provides pressure to move fluids."
        elif "Tank" in type_clean or "Vessel" in type_clean:
            role_desc = " It stores fluid inventory."
        elif "HeatExchanger" in type_clean:
            role_desc = " It transfers thermal energy between fluids."

        base_desc = f"{prefix}In DEXPI file '{filename}', entity '{tag_clean}' (ID: {item['id']}) is classified as a '{type_clean}'.{role_desc}"
        
        attr_sentences = []
        if item.get('attributes'):
            for key, value in item['attributes'].items():
                if "AssignmentClass" in key:
                    clean_key = key.replace("AssignmentClass", "").replace("Code", "")
                    attr_sentences.append(f"Its {clean_key} is '{value}'.")
                elif "Specialization" in key:
                    clean_key = key.replace("Specialization", "")
                    attr_sentences.append(f"Its {clean_key} is '{value}'.")
                else:
                    attr_sentences.append(f"Its {key} is '{value}'.")
        
        full_paragraph = f"{base_desc} {' '.join(attr_sentences)}"
        texts_to_ingest.append(full_paragraph)

    # --- 2B. Connections ---
    id_map = {item['id']: {"tag": item.get('tag', 'Unknown'), "type": item.get('type', 'Unknown')} 
              for item in parsed_data['equipment']}

    for conn in parsed_data['connections']:
        src_id = conn['source']
        tgt_id = conn['target']
        
        if src_id in id_map and tgt_id in id_map:
            src_info = id_map[src_id]
            tgt_info = id_map[tgt_id]
            
            rel_text = (
                f"{prefix}Connection in '{filename}': "
                f"The {src_info['type']} '{src_info['tag']}' (ID: {src_id}) "
                f"is connected to "
                f"the {tgt_info['type']} '{tgt_info['tag']}' (ID: {tgt_id})."
            )
            texts_to_ingest.append(rel_text)

    with open(f"{DEBUG_DIR}/2_synthetic_input.txt", "w") as f:
        f.write("\n".join(texts_to_ingest))
    
    print(f"Saved synthetic input to '2_synthetic_input.txt'")

    # =========================================================================
    # STEP 3: INGESTION
    # =========================================================================
    print("\n--- STEP 3: INGESTION ---")
    engine = LightRAGEngine()
    await engine.initialize()
    
    print("Ingesting data into LightRAG...")
    await engine.rag.ainsert(texts_to_ingest)
    print("Ingestion Complete.")

    # =========================================================================
    # STEP 4: CUSTOM RETRIEVAL (Bypassing LightRAG's query)
    # =========================================================================
    print("\n--- STEP 4: CUSTOM RETRIEVAL ---")
    query = "What is Wirkungslinie classified as?"
    
    # CALL CUSTOM FUNCTION HERE
    context = await custom_retriever(query)
    
    with open(f"{DEBUG_DIR}/4_retrieved_context.txt", "w") as f:
        f.write(context)
        
    print(f"Saved Retrieved Context to '4_retrieved_context.txt'")
    print("CONTEXT PREVIEW:")
    print(f"   {context[:]}...")


    # =========================================================================
    # STEP 5: FINAL GENERATION
    # =========================================================================
    print("\n--- STEP 5: FINAL GENERATION ---")
    
    prompt = f"""
    Based ONLY on the context below, answer the question given by the user.
    If the answer is in the context, repeat the definition exactly.
    
    CONTEXT:
    {context}
    
    QUESTION: {query}
    
    ANSWER:
    """
    
    answer = await ollama_llm_func(prompt)
    
    with open(f"{DEBUG_DIR}/5_final_answer.txt", "w") as f:
        f.write(answer)
        
    print(f"Saved Final Answer to '5_final_answer.txt'")
    print("\n FINAL ANSWER:")
    print("-" * 50)
    print(answer)
    print("-" * 50)

if __name__ == "__main__":
    asyncio.run(main())