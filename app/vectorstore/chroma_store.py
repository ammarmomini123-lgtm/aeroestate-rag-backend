import os
import uuid
from typing import List, Dict, Any, Union, Optional
import numpy as np
import chromadb
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from app.config.settings import settings


class ChromaVectorStore:
    """Manages persistent document storage, metadata retrieval, updates, and deletion using ChromaDB."""

    def __init__(self, collection_name: str = "pdf_rag_collection", persist_dir: str = getattr(settings, "VECTOR_DB_DIR", "./chroma_db")):
        os.makedirs(persist_dir, exist_ok=True)
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def _to_list(self, vector: Union[np.ndarray, List]) -> List:
        """Helper to ensure vector is a standard Python list."""
        if isinstance(vector, np.ndarray):
            return vector.tolist()
        return vector

    def add_documents(
        self, 
        chunks: List[Document], 
        embeddings: Union[np.ndarray, List[List[float]]], 
        batch_size: int = 250
    ) -> None:
        """Inserts document chunks and embeddings into ChromaDB using safe batches."""
        if not chunks:
            print("⚠️ No chunks to insert into ChromaDB.")
            return

        embeddings_list = self._to_list(embeddings)

        if len(chunks) != len(embeddings_list):
            raise ValueError(f"Mismatch: {len(chunks)} chunks vs {len(embeddings_list)} embeddings.")

        total_chunks = len(chunks)

        for i in range(0, total_chunks, batch_size):
            batch_chunks = chunks[i : i + batch_size]
            batch_embeddings = embeddings_list[i : i + batch_size]

            batch_ids = [
                str(chunk.metadata.get("chunk_id")) if chunk.metadata.get("chunk_id") 
                else f"{uuid.uuid4().hex[:8]}_{i + idx}" 
                for idx, chunk in enumerate(batch_chunks)
            ]
            batch_texts = [chunk.page_content for chunk in batch_chunks]
            batch_metadatas = [
                {
                    "document_id": str(chunk.metadata.get("document_id") or "doc_unknown"),
                    "file_name": str(chunk.metadata.get("file_name") or "Unknown Document"),
                    "file_path": str(chunk.metadata.get("file_path") or "Unknown Path"),
                    "file_type": str(chunk.metadata.get("file_type") or "pdf"),
                    "author": str(chunk.metadata.get("author") or "Unknown Author"),
                    "page": int(chunk.metadata.get("page") or 1)
                }
                for chunk in batch_chunks
            ]

            self.collection.upsert(
                ids=batch_ids,
                documents=batch_texts,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas
            )

        print(f"✓ Saved {total_chunks} chunks to ChromaDB.")

    def search(
        self, 
        query_embedding: Union[np.ndarray, List[float]], 
        top_k: int = getattr(settings, "TOP_K", 4),
        where_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Queries ChromaDB using cosine similarity with optional metadata filtering."""
        query_list = self._to_list(query_embedding)

        if query_list and not isinstance(query_list[0], list):
            query_list = [query_list]

        query_kwargs = {
            "query_embeddings": query_list,
            "n_results": top_k
        }
        if where_filter:
            query_kwargs["where"] = where_filter

        results = self.collection.query(**query_kwargs)

        formatted_results = []
        if results and results.get('documents') and len(results['documents']) > 0:
            docs = results['documents'][0]
            metas = results['metadatas'][0]
            distances = results['distances'][0] if 'distances' in results and results['distances'] else [0.0] * len(docs)
            ids = results['ids'][0] if 'ids' in results and results['ids'] else [""] * len(docs)

            for doc_id, doc, meta, dist in zip(ids, docs, metas, distances):
                similarity_score = max(0.0, 1.0 - float(dist))
                formatted_results.append({
                    "chunk_id": doc_id,
                    "content": doc,
                    "document_id": meta.get("document_id", "doc_unknown"),
                    "file_name": meta.get("file_name", "Unknown Document"),
                    "file_path": meta.get("file_path", "Unknown Path"),
                    "file_type": meta.get("file_type", "pdf"),
                    "author": meta.get("author", "Unknown Author"),
                    "page": meta.get("page", 1),
                    "score": similarity_score
                })

        return formatted_results

    def delete_by_file_name(self, file_name: str) -> None:
        """Deletes all chunks belonging to a specific file name."""
        self.collection.delete(where={"file_name": file_name})
        print(f"✓ Deleted all chunks associated with file '{file_name}' from ChromaDB.")

    def delete_by_document_id(self, document_id: str) -> None:
        """Deletes all chunks belonging to a specific document ID."""
        self.collection.delete(where={"document_id": document_id})
        print(f"✓ Deleted all chunks associated with document_id '{document_id}' from ChromaDB.")

    def get_stats(self) -> Dict[str, Any]:
        """Returns statistics for the dashboard UI."""
        all_data = self.collection.get(include=["metadatas"])
        total_chunks = len(all_data.get("ids", []))
        metadatas = all_data.get("metadatas", [])
        
        distinct_files = set(m.get("file_name") for m in metadatas if m and "file_name" in m)
        
        return {
            "total_indexed_chunks": total_chunks,
            "total_documents": len(distinct_files),
            "indexed_files": list(distinct_files)
        }

    def clear_store(self) -> None:
        """Drops and recreates the collection to wipe all data cleanly."""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"✓ Cleared all entries from '{self.collection_name}'.")


class ChromaCustomRetriever(BaseRetriever):
    vector_store: ChromaVectorStore
    embedder: Any
    where_filter: Optional[Dict[str, Any]] = None

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        query_embedding = self.embedder.embed_query(query)
        results = self.vector_store.search(
            query_embedding, 
            top_k=getattr(settings, "TOP_K", 4),
            where_filter=self.where_filter
        )

        return [
            Document(
                page_content=res["content"],
                metadata={
                    "chunk_id": res["chunk_id"],
                    "document_id": res["document_id"],
                    "file_name": res["file_name"],
                    "file_path": res["file_path"],
                    "file_type": res["file_type"],
                    "author": res["author"],
                    "page": res["page"],
                    "score": res["score"]
                }
            )
            for res in results
        ]