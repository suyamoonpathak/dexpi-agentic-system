import os
import json
import base64
import zlib
import numpy as np

KB_DIR = "data/knowledge_base"
EXPECTED_DIM = 768

def recursive_find_first_vector(data):
    """Finds the first dictionary containing a 'vector' key."""
    if isinstance(data, dict):
        if "vector" in data and isinstance(data["vector"], str): 
            return data["vector"]
        for k, v in data.items():
            res = recursive_find_first_vector(v)
            if res: return res
    elif isinstance(data, list):
        for item in data:
            res = recursive_find_first_vector(item)
            if res: return res
    return None

def test_format(name, func, vec_str):
    print(f"\n TESTING FORMAT: {name}")
    try:
        vec = func(vec_str)
        shape = vec.shape
        print(f"   Shape: {shape}")
        
        if shape == (EXPECTED_DIM,):
            print(f" SUCCESS! This is the correct format.")
            return True, vec
        elif shape[0] * 2 == EXPECTED_DIM:
            print(f" Dimension is half ({shape[0]}). Might need Float16?")
        elif shape[0] / 2 == EXPECTED_DIM:
            print(f" Dimension is double ({shape[0]}). Might be Float64 interpreted as Float32?")
        else:
            print(f"Incorrect dimension.")
    except Exception as e:
        print(f"CRASHED: {e}")
    return False, None

#  DECODERS TO TEST 

def decode_b64_float32(s):
    b = base64.b64decode(s)
    return np.frombuffer(b, dtype=np.float32)

def decode_b64_float64(s):
    b = base64.b64decode(s)
    return np.frombuffer(b, dtype=np.float64)

def decode_b64_zlib_float32(s):
    b = base64.b64decode(s)
    b = zlib.decompress(b)
    return np.frombuffer(b, dtype=np.float32)

def decode_b64_zlib_float64(s):
    b = base64.b64decode(s)
    b = zlib.decompress(b)
    return np.frombuffer(b, dtype=np.float64)

def main():
    vdb_path = os.path.join(KB_DIR, "vdb_chunks.json")
    
    if not os.path.exists(vdb_path):
        print("File not found.")
        return

    with open(vdb_path, "r") as f:
        data = json.load(f)

    vec_str = recursive_find_first_vector(data)
    
    if not vec_str:
        print("Could not find any vector string to test.")
        return

    print(f" Found sample vector string. Length: {len(vec_str)} chars")
    
    tests = [
        ("Base64 -> Float32 (Standard)", decode_b64_float32),
        ("Base64 -> Float64 (High Precision)", decode_b64_float64),
        ("Base64 -> Zlib -> Float32 (Compressed)", decode_b64_zlib_float32),
        ("Base64 -> Zlib -> Float64 (Compressed High Precision)", decode_b64_zlib_float64),
    ]

    for name, func in tests:
        success, _ = test_format(name, func, vec_str)
        if success:
            print("\n FORMAT IDENTIFIED.")
            break

if __name__ == "__main__":
    main()