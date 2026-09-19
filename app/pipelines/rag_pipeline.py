from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from app.llm.llm import LLMProvider
from app.vectorstore.chroma_store import ChromaVectorStore, ChromaCustomRetriever
from app.embeddings.embedding_model import EmbeddingModel 
from app.config.settings import settings


class RAGPipeline:
    """Combines context retrieval with strict LLM answer generation and citation extraction."""

    def __init__(self):
        self.vector_store = ChromaVectorStore()
        self.embedder = EmbeddingModel()
        
        self.retriever = ChromaCustomRetriever(
            vector_store=self.vector_store,
            embedder=self.embedder
        )

        self.llm = LLMProvider().get_llm()

        # Updated prompt tailored for clean email auto-responses
        self.prompt = ChatPromptTemplate.from_template(
            """You are a helpful and polite customer support representative for AeroEstate Realty.
Answer the user's question clearly and concisely using ONLY the provided context chunks.

CONTEXT:
{context}

QUESTION:
{question}

INSTRUCTIONS & FORMATTING RULES:
1. Ground your answer strictly in the provided context above. Do not guess, speculate, or draw on outside general knowledge.
2. If the requested information cannot be found in the provided context, state clearly: "The requested information could not be found in our knowledge base."
3. Write in a clear, professional email tone. Use proper bullet points and paragraphs for readability.
4. DO NOT include email headers, subjects, greetings (like "Dear [Client Name]"), or sign-offs (like "Best regards, [Your Name]") inside the generated answer text. Focus ONLY on providing the clear summary/body text.
5. DO NOT include raw source citations, file names (e.g., 📄 Document.pdf), page numbers, or horizontal rule separators (---) in your answer text. Keep the body clean and client-facing."""
        )

    def run(self, question: str) -> Dict[str, Any]:
        """Executes RAG pipeline: Retrieve -> Guardrail Check -> LLM Generate -> Extract Sources."""
        if not question or not question.strip():
            return {
                "answer": "Please provide a valid question.",
                "context": []
            }

        # 1. Retrieve relevant chunks
        docs = self.retriever.invoke(question)

        # 2. Early refusal if no chunks pass threshold
        if not docs:
            return {
                "answer": "The requested information could not be found in our knowledge base.",
                "context": []
            }

        # 3. Format context string with explicit metadata headers
        context_parts = []
        for idx, doc in enumerate(docs, start=1):
            file_name = doc.metadata.get("file_name", "Unknown File")
            page = doc.metadata.get("page", 1)
            context_parts.append(
                f"[Chunk {idx} | File: {file_name} | Page: {page}]\n{doc.page_content}"
            )
        context_str = "\n\n---\n\n".join(context_parts)

        # 4. Generate answer via LLM
        chain = self.prompt | self.llm
        response = chain.invoke({"context": context_str, "question": question})

        # Parse string safely
        raw_content = response.content
        if isinstance(raw_content, list):
            text_parts = []
            for item in raw_content:
                if isinstance(item, dict) and "text" in item:
                    text_parts.append(item["text"])
                elif isinstance(item, str):
                    text_parts.append(item)
            answer_text = "".join(text_parts).strip()
        else:
            answer_text = str(raw_content).strip()

        # 5. Hallucination Guardrail Refusal Check
        fallback_phrases = [
            "could not be found",
            "not contain information",
            "does not contain",
            "information is not available",
            "cannot answer",
            "not mentioned in the provided"
        ]

        is_refusal = any(phrase in answer_text.lower() for phrase in fallback_phrases)

        # 6. Build Deduplicated Source Citations (Saved for backend context logging)
        formatted_sources: List[Dict[str, Any]] = []
        seen_sources = set()

        if not is_refusal:
            for doc in docs:
                file_name = doc.metadata.get("file_name", "Unknown File")
                page = doc.metadata.get("page", 1)
                doc_id = doc.metadata.get("document_id", "doc_gen")
                chunk_id = doc.metadata.get("chunk_id", "")

                source_key = f"{file_name}_p{page}"
                if source_key not in seen_sources:
                    seen_sources.add(source_key)
                    formatted_sources.append({
                        "document_id": doc_id,
                        "chunk_id": chunk_id,
                        "file_name": file_name,
                        "file_path": doc.metadata.get("file_path", "N/A"),
                        "file_type": doc.metadata.get("file_type", "pdf"),
                        "author": doc.metadata.get("author", "Unknown Author"),
                        "page": page,
                        "score": round(float(doc.metadata.get("score", 0.0)), 4),
                        "content_snippet": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                    })

        return {
            "answer": answer_text,
            "context": formatted_sources
        }