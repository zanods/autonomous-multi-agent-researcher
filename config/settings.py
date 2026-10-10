from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    google_api_key: str = ""
    groq_api_key: str = ""
    cerebras_api_key: str = ""
    openrouter_api_key: str = ""
    openai_api_key: str = ""
    model: str = "gemini-3.8-flash"
    embedding_model: str = "text-embedding-3-small"
    tavily_api_key: str = ""
    cohere_api_key: str = ""
    chroma_dir: str = "./chroma_db"
    optimize_tokens: bool = True

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
