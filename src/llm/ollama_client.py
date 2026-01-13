import os
import numpy as np
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from config.settings import get_settings
from src.rag.prompts import GRAPH_EXTRACTION_PROMPT

settings = get_settings()

client = AsyncOpenAI(
    base_url=f"http://{settings.OLLAMA_SERVER_IP}:{settings.OLLAMA_PORT}/v1",
    api_key="ollama", 
)

class OllamaClient:
    def __init__(self):
        self.model = settings.OLLAMA_MODEL
        self.embed_model = settings.OLLAMA_EMBED_MODEL

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3), 
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def agenerate(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        """
        Generates text using the remote Ollama model.
        """
        if system_prompt and "entity" in system_prompt and "relationship" in system_prompt:
            system_prompt = GRAPH_EXTRACTION_PROMPT

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Ollama crashes on unknown args passed by LightRAG. So, stripping them.
        keys_to_remove = [
            "hashing_kv", "keyword_fields", "history_messages", "mode", "original_query",
            "keyword_extraction", "language", "enable_cot", "mix_mode"
        ]
        for key in keys_to_remove:
            kwargs.pop(key, None)
        
        if "max_tokens" not in kwargs:
            kwargs["max_tokens"] = 4096

        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                **kwargs 
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[Ollama Error] Generation failed: {str(e)}")
            raise e

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3), 
        wait=wait_exponential(multiplier=1, min=1, max=5)
    )
    async def aembed(self, texts: list[str]) -> np.ndarray:
        try:
            cleaned_texts = [t.replace("\n", " ") for t in texts]
            
            response = await client.embeddings.create(
                model=self.embed_model,
                input=cleaned_texts
            )
            return np.array([item.embedding for item in response.data])
        except Exception as e:
            print(f"[Ollama Error] Embedding failed: {str(e)}")
            raise e

# Adapter Functions
async def ollama_llm_func(prompt: str, system_prompt: str = None, **kwargs) -> str:
    client_instance = OllamaClient()
    return await client_instance.agenerate(prompt, system_prompt, **kwargs)

async def ollama_embedding_func(texts: list[str]) -> np.ndarray:
    client_instance = OllamaClient()
    return await client_instance.aembed(texts)