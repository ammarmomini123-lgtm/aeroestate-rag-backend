import os
import re
import uuid
from typing import List, Optional
from pypdf import PdfReader
import docx2txt
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config.settings import settings


class DocumentIngestionPipeline:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )

    def extract_from_pdf(self, file_path: str, doc_id: str) -> List[Document]:
        """Extracts text page-by-page from PDF with metadata."""
        documents = []
        reader = PdfReader(file_path)
        file_name = os.path.basename(file_path)

        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            cleaned_text = re.sub(r"\s+", " ", text).strip()

            if len(cleaned_text) < 20:
                continue

            doc = Document(
                page_content=cleaned_text,
                metadata={
                    "document_id": doc_id,
                    "file_name": file_name,
                    "file_path": file_path,
                    "file_type": "pdf",
                    "author": reader.metadata.author if reader.metadata and reader.metadata.author else "Unknown",
                    "page": page_num,
                }
            )
            documents.append(doc)

        return documents

    def extract_from_docx(self, file_path: str, doc_id: str) -> List[Document]:
        """Extracts text from DOCX files."""
        text = docx2txt.process(file_path) or ""
        cleaned_text = re.sub(r"\s+", " ", text).strip()
        
        if len(cleaned_text) < 20:
            return []

        return [
            Document(
                page_content=cleaned_text,
                metadata={
                    "document_id": doc_id,
                    "file_name": os.path.basename(file_path),
                    "file_path": file_path,
                    "file_type": "docx",
                    "author": "Unknown",
                    "page": 1,  # DOCX lacks explicit pages; default to 1
                }
            )
        ]

    def extract_from_txt(self, file_path: str, doc_id: str) -> List[Document]:
        """Extracts text from raw TXT files."""
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
            
        cleaned_text = re.sub(r"\s+", " ", text).strip()
        if len(cleaned_text) < 20:
            return []

        return [
            Document(
                page_content=cleaned_text,
                metadata={
                    "document_id": doc_id,
                    "file_name": os.path.basename(file_path),
                    "file_path": file_path,
                    "file_type": "txt",
                    "author": "Unknown",
                    "page": 1,
                }
            )
        ]

    def process_file(self, file_path: str, custom_doc_id: Optional[str] = None) -> List[Document]:
        """Processes a single file (PDF, DOCX, TXT), splits into chunks, and assigns chunk IDs."""
        ext = os.path.splitext(file_path)[1].lower()
        doc_id = custom_doc_id or str(uuid.uuid4())[:8]

        if ext == ".pdf":
            raw_docs = self.extract_from_pdf(file_path, doc_id)
        elif ext == ".docx":
            raw_docs = self.extract_from_docx(file_path, doc_id)
        elif ext in [".txt", ".md"]:
            raw_docs = self.extract_from_txt(file_path, doc_id)
        else:
            print(f"Unsupported file format: {ext}")
            return []

        if not raw_docs:
            return []

        chunks = self.text_splitter.split_documents(raw_docs)
        
        # Filter tiny residual chunks and inject unique chunk IDs
        valid_chunks = []
        for idx, chunk in enumerate(chunks):
            if len(chunk.page_content.strip()) > 30:
                chunk.metadata["chunk_id"] = f"{doc_id}_c{idx+1}"
                valid_chunks.append(chunk)

        return valid_chunks

    def process_all_documents(self) -> List[Document]:
        """Processes all PDF, DOCX, and TXT files from RAW_DATA_DIR."""
        all_chunks = []
        raw_dir = settings.RAW_DATA_DIR

        if not os.path.exists(raw_dir):
            print(f"Directory {raw_dir} does not exist.")
            return []

        for root, _, files in os.walk(raw_dir):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in [".pdf", ".docx", ".txt", ".md"]:
                    full_path = os.path.join(root, file)
                    print(f"Processing: {file}")
                    file_chunks = self.process_file(full_path)
                    all_chunks.extend(file_chunks)

        print(f"Total valid chunks generated across all files: {len(all_chunks)}")
        return all_chunks