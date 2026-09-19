import os
import uuid
from typing import Optional, List
from app.loaders.document_loader import DocumentLoader
from app.embeddings.embedding_model import EmbeddingModel
from app.vectorstore.chroma_store import ChromaVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config.settings import settings


class IngestionPipeline:
    """Handles automatic discovery, loading, chunking, embedding, and vector storage for PDF, DOCX, and TXT files."""

    def __init__(self):
        self.loader = DocumentLoader()
        self.embedding_model = EmbeddingModel()
        self.vector_store = ChromaVectorStore()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=getattr(settings, "CHUNK_SIZE", 700),
            chunk_overlap=getattr(settings, "CHUNK_OVERLAP", 100),
            separators=["\n\n", "\n", " ", ""]
        )

    def run(self, source_path: Optional[str] = None) -> bool:
        path_to_process = source_path or getattr(settings, "RAW_DATA_DIR", "./data/raw")
        print(f"\n--- Starting Ingestion Pipeline for: '{path_to_process}' ---")

        # 1. Load documents (Single file or directory scan)
        if os.path.isfile(path_to_process):
            doc_id = str(uuid.uuid4())[:8]
            documents = self.loader.load_single_file(path_to_process, doc_id=doc_id)
        elif os.path.isdir(path_to_process):
            documents = self.loader.load_directory(path_to_process)
        else:
            print(f"❌ Error: Invalid path specified: {path_to_process}")
            return False

        if not documents:
            print("--- Ingestion Skipped (No valid document content loaded) ---")
            return False

        # 2. Chunk documents into chunks
        chunks = self.text_splitter.split_documents(documents)
        
        # Inject chunk_id and preserve metadata
        valid_chunks = []
        for idx, chunk in enumerate(chunks):
            content = chunk.page_content.strip()
            if len(content) > 30:
                doc_id = chunk.metadata.get("document_id", "doc_gen")
                chunk.metadata["chunk_id"] = f"{doc_id}_c{idx+1}"
                valid_chunks.append(chunk)

        if not valid_chunks:
            print("--- Ingestion Skipped (No valid chunks created after filtering) ---")
            return False

        print(f"✓ Split {len(documents)} document pages/units into {len(valid_chunks)} chunks.")

        # 3. Generate embeddings
        texts = [chunk.page_content for chunk in valid_chunks]
        embeddings = self.embedding_model.embed_documents(texts)

        # 4. Save to ChromaDB
        self.vector_store.add_documents(valid_chunks, embeddings)
        print("--- Ingestion Pipeline Finished Successfully ---")
        return True