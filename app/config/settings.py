import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # API Keys (support both GEMINI_API_KEY and GOOGLE_API_KEY)
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "openai/gpt-oss-120b")

    # Directories
    RAW_DATA_DIR: str = os.getenv("RAW_DATA_DIR", "data/raw")
    VECTOR_DB_DIR: str = os.getenv("VECTOR_DB_DIR", "data/vector_db")

    # Ingestion Parameters
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "700"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))

    # Models & Retrieval
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "gemini-1.5-flash")
    TOP_K: int = int(os.getenv("TOP_K", "4"))


settings = Settings()