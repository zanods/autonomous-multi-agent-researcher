from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str = ""
    model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    tavily_api_key: str = ""
    cohere_api_key: str = ""
    chroma_dir: str = "./chroma_db"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
