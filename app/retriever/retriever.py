from typing import List, Dict, Any, Optional
from app.vectorstore.chroma_store import ChromaVectorStore
from app.embeddings.embedding_model import EmbeddingModel


class DocumentRetriever:
    """Retrieves relevant chunks from ChromaDB for a given user query."""

    def __init__(self, vector_store: ChromaVectorStore, embedding_model: EmbeddingModel):
        self.vector_store = vector_store
        self.embedding_model = embedding_model

    def get_relevant_chunks(
        self, 
        query: str, 
        top_k: int = 4, 
        min_score: float = 0.25,
        where_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Embeds the input query, queries ChromaDB, and filters out low-relevance results.
        """
        if not query or not query.strip():
            return []

        # 1. Embed user query
        query_embedding = self.embedding_model.embed_query(query)
        if not query_embedding:
            return []

        # 2. Perform vector search in ChromaDB
        results = self.vector_store.search(
            query_embedding, 
            top_k=top_k, 
            where_filter=where_filter
        )

        # 3. Format and filter by minimum similarity score threshold
        formatted_results = []
        for res in results:
            score = float(res.get("score", 0.0))

            # Filter out chunks that do not meet the minimum similarity threshold
            if score >= min_score:
                formatted_results.append({
                    "chunk_id": res.get("chunk_id", ""),
                    "document_id": res.get("document_id", "doc_unknown"),
                    "content": res.get("content", ""),
                    "file_name": res.get("file_name", "Unknown Document"),
                    "file_path": res.get("file_path", ""),
                    "file_type": res.get("file_type", "pdf"),
                    "author": res.get("author", "Unknown Author"),
                    "page": res.get("page", 1),
                    "score": round(score, 4)
                })

        return formatted_results