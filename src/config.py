from pathlib import Path
from pydantic_settings import BaseSettings

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE_PATH = PROJECT_ROOT / ".env"

class Settings(BaseSettings):
    qdrant_url: str
    qdrant_api_key: str
    collection_name: str = "egyptian_civil_code"

    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024

    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_top_k: int = 3

    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"

    # MLflow tracking URI (optional)
    mlflow_tracking_uri: str | None = None

    class Config:
        env_file = str(ENV_FILE_PATH)

settings = Settings()