import os
import json
import asyncio
import base64
import zlib
import numpy as np
from src.llm.ollama_client import ollama_embedding_func

KB_DIR = "data/knowledge_base"
QUERY = "What is Wirkungslinie classified as?"
TARGET_KEYWORD = "Wirkungslinie"

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
            
            # 3. Float16
            # We convert it up to Float32 immediately for better math precision later
            return np.frombuffer(vec_bytes, dtype=np.float16).astype(np.float32)
        except Exception as e:
            return np.zeros(1)
            
    return np.zeros(1)

def recursive_find_chunks(data, chunks_list):
    if isinstance(data, dict):
        if "vector" in data and "content" in data: 
            chunks_list.append(data)
        for k, v in data.items():
            recursive_find_chunks(v, chunks_list)
    elif isinstance(data, list):
        for item in data:
            recursive_find_chunks(item, chunks_list)

def cosine_similarity(v1, v2):
    if v1.shape != v2.shape: return -1.0
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return 0.0 if (norm_v1 == 0 or norm_v2 == 0) else np.dot(v1, v2) / (norm_v1 * norm_v2)

async def diagnose():
    print("STARTING RETRIEVAL DIAGNOSIS (FLOAT16 SUPPORT)")
    print("-" * 50)

    # 1. LOAD
    vdb_path = os.path.join(KB_DIR, "vdb_chunks.json")
    with open(vdb_path, "r") as f:
        raw_json = json.load(f)

    # 2. FIND
    chunks = []
    recursive_find_chunks(raw_json, chunks)
    print(f"Found {len(chunks)} chunks.")

    # 3. EMBED
    print("Embedding query...")
    query_vec = await ollama_embedding_func([QUERY])
    query_vec = np.array(query_vec[0], dtype=np.float32)
    print(f"   Query Shape: {query_vec.shape}")

    # 4. RANK
    scores = []
    
    for ch in chunks:
        vec = decode_vector(ch["vector"])
        
        score = cosine_similarity(query_vec, vec)
        
        scores.append({
            "score": score,
            "content": ch.get("content", ""),
            "is_target": TARGET_KEYWORD in ch.get("content", "")
        })

    scores.sort(key=lambda x: x['score'], reverse=True)

    # 5. REPORT
    target_rank = -1
    for i, item in enumerate(scores):
        if item['is_target']:
            target_rank = i + 1
            print(f"\nTARGET FOUND AT RANK #{target_rank}")
            print(f"   Score: {item['score']:.4f}")
            print(f"   Content: {item['content'][:]}...")
            break
            
    if target_rank == -1:
        print("\nTarget text NOT found.")
    elif target_rank <= 5:
        print(f"\nSUCCESS: Data is at Rank #{target_rank}.")

    else:
        print(f"\nPOOR RANKING (Rank #{target_rank})")

        for i in range(min(3, len(scores))):
             print(f"   #{i+1}: {scores[i]['score']:.4f} - {scores[i]['content'][:]}...")

if __name__ == "__main__":
    asyncio.run(diagnose())