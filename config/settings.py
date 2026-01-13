from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    OLLAMA_SERVER_IP: str
    OLLAMA_PORT: int = 11435
    OLLAMA_MODEL: str = "llama3.3:7b"
    OLLAMA_EMBED_MODEL: str = "nomic-embed-text"
    
    DATA_DIR: str = "data"
    WORKING_DIR: str = "data/knowledge_base"

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        extra='ignore' 
    )

@lru_cache()
def get_settings():
    return Settings()