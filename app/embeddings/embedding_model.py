from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from app.config.settings import settings


class EmbeddingModel(Embeddings):
    """
    Handles vector embedding generation using SentenceTransformers.
    Implements LangChain's Embeddings interface for seamless Vector Store integration.
    """

    def __init__(self, model_name: str = getattr(settings, "EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generates vector embeddings for a list of document chunk texts.
        """
        if not texts:
            return []
        
        # Batch encode with normalization for accurate cosine similarity search
        embeddings = self.model.encode(
            texts, 
            batch_size=32, 
            show_progress_bar=False, 
            normalize_embeddings=True
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """
        Generates vector embedding for a single search query string.
        """
        if not text or not text.strip():
            return []
            
        embedding = self.model.encode(
            text, 
            normalize_embeddings=True
        )
        return embedding.tolist()