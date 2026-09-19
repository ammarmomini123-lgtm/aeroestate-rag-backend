from typing import List, Dict, Any, Optional
from app.vectorstore.chroma_store import ChromaVectorStore
from app.embeddings.embedding_model import EmbeddingModel
from app.retriever.retriever import DocumentRetriever
from app.config.settings import settings


class RetrievalPipeline:
    """Standalone retrieval pipeline that embeds a query and returns relevant context chunks with metadata."""

    def __init__(self):
        self.embedding_model = EmbeddingModel()
        self.vector_store = ChromaVectorStore()
        self.retriever = DocumentRetriever(self.vector_store, self.embedding_model)

    def run(
        self, 
        query: str, 
        top_k: int = getattr(settings, "TOP_K", 4),
        min_score: float = 0.25,
        where_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes semantic search against ChromaDB and formats chunks with source citations.
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            print("⚠️ Warning: Empty query provided to RetrievalPipeline.")
            return []

        print(f"🔎 Running retrieval query: '{cleaned_query}' (top_k={top_k}, min_score={min_score})")

        # Query retriever
        chunks = self.retriever.get_relevant_chunks(
            query=cleaned_query, 
            top_k=top_k, 
            min_score=min_score,
            where_filter=where_filter
        )

        if not chunks:
            print("⚠️ Warning: No relevant chunks met the similarity threshold.")
        else:
            print(f"✓ Retrieved {len(chunks)} relevant chunk(s) from vector store.")

        return chunks