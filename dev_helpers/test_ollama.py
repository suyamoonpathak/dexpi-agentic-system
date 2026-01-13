import sys
import os
import asyncio
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.llm.ollama_client import ollama_llm_func, ollama_embedding_func
from config.settings import get_settings

settings = get_settings()

async def main():
    print(f"Testing Connection to Ollama at: http://{settings.OLLAMA_SERVER_IP}:{settings.OLLAMA_PORT}")
    print(f"Target Gen Model: {settings.OLLAMA_MODEL}")
    print(f"Target Embed Model: {settings.OLLAMA_EMBED_MODEL}")
    print("-" * 50)

    # 1. Test Text Generation
    print("\n1. Testing Text Generation...")
    try:
        response = await ollama_llm_func("Hello, are you ready for the DEXPI assignment?")
        print(f"Success! Response:\n{response}")
    except Exception as e:
        print(f"Generation Failed: {e}")

    # 2. Test Embeddings
    print("\n2. Testing Embeddings...")
    try:
        test_text = ["Pump P-101", "Tank T-500"]
        embeddings = await ollama_embedding_func(test_text)
        
        if isinstance(embeddings, np.ndarray) and embeddings.shape[0] == 2:
            print(f"Success! Embedding Shape: {embeddings.shape}")
        else:
            print(f" Unexpected Output format: {type(embeddings)}")
    except Exception as e:
        print(f" Embedding Failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())