import os
import re
import uuid
import zipfile
import xml.etree.ElementTree as ET
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


class DocumentLoader:
    """Unified loader for PDF, DOCX, and TXT files with standardized metadata."""

    def extract_text_from_docx(self, file_path: str) -> str:
        """Extracts text directly from DOCX XML structure without external dependencies."""
        try:
            with zipfile.ZipFile(file_path) as z:
                xml_content = z.read("word/document.xml")
            tree = ET.fromstring(xml_content)
            
            # W3C namespace for Word XML
            ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
            texts = []
            
            for p in tree.findall(".//w:p", ns):
                paragraph_text = "".join([node.text for node in p.findall(".//w:t", ns) if node.text])
                if paragraph_text.strip():
                    texts.append(paragraph_text)
                    
            return "\n".join(texts)
        except Exception as e:
            print(f"Error parsing DOCX file {file_path}: {e}")
            return ""

    def load_single_file(self, file_path: str, doc_id: str = None) -> List[Document]:
        """Loads a single PDF, DOCX, or TXT file and attaches RAG metadata."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_name = os.path.basename(file_path)
        ext = os.path.splitext(file_name)[1].lower()
        document_id = doc_id or str(uuid.uuid4())[:8]
        valid_documents = []

        if ext == ".pdf":
            try:
                loader = PyPDFLoader(file_path)
                raw_docs = loader.load()
                for doc in raw_docs:
                    cleaned_text = re.sub(r"\s+", " ", doc.page_content).strip()
                    if len(cleaned_text) >= 20:
                        doc.page_content = cleaned_text
                        doc.metadata.update({
                            "document_id": document_id,
                            "file_name": file_name,
                            "file_path": file_path,
                            "file_type": "pdf",
                            "page": doc.metadata.get("page", 1) + 1  # 1-indexed page
                        })
                        valid_documents.append(doc)
            except Exception as e:
                print(f"Error reading PDF {file_name}: {e}")

        elif ext == ".docx":
            raw_text = self.extract_text_from_docx(file_path)
            cleaned_text = re.sub(r"\s+", " ", raw_text).strip()
            if len(cleaned_text) >= 20:
                doc = Document(
                    page_content=cleaned_text,
                    metadata={
                        "document_id": document_id,
                        "file_name": file_name,
                        "file_path": file_path,
                        "file_type": "docx",
                        "page": 1,
                    }
                )
                valid_documents.append(doc)

        elif ext in [".txt", ".md"]:
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    raw_text = f.read()
                cleaned_text = re.sub(r"\s+", " ", raw_text).strip()
                if len(cleaned_text) >= 20:
                    doc = Document(
                        page_content=cleaned_text,
                        metadata={
                            "document_id": document_id,
                            "file_name": file_name,
                            "file_path": file_path,
                            "file_type": "txt",
                            "page": 1,
                        }
                    )
                    valid_documents.append(doc)
            except Exception as e:
                print(f"Error reading text file {file_name}: {e}")

        else:
            print(f"⚠️ Unsupported file type: {ext}")

        print(f"✓ Loaded {len(valid_documents)} document unit(s) from '{file_name}'.")
        return valid_documents

    def load_directory(self, folder_path: str) -> List[Document]:
        """Scans a folder and loads all PDF, DOCX, and TXT files."""
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Directory not found: {folder_path}")

        supported_exts = (".pdf", ".docx", ".txt", ".md")
        all_documents = []

        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(supported_exts):
                    file_path = os.path.join(root, file)
                    docs = self.load_single_file(file_path)
                    all_documents.extend(docs)

        return all_documents


# Alias for backward compatibility
PDFLoader = DocumentLoader