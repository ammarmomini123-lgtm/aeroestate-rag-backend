from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Any

app = FastAPI(
    title="AeroEstate RAG API",
    description="Backend API for Task 6 n8n Email Triage Integration",
    version="1.0.0"
)

# Pydantic Schemas
class QueryRequest(BaseModel):
    question: str
    email_sender: Optional[str] = None
    email_subject: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    context: Optional[List[str]] = []
    status: str


# Global instance for lazy loading RAG Pipeline
rag_pipeline_instance = None


def get_rag_pipeline():
    global rag_pipeline_instance
    if rag_pipeline_instance is None:
        print("⏳ Lazy loading RAG Pipeline into memory...")
        from app.pipelines.rag_pipeline import RAGPipeline
        rag_pipeline_instance = RAGPipeline()
        print("✅ RAG Pipeline loaded successfully.")
    return rag_pipeline_instance


@app.get("/")
def health_check():
    # Returns 200 OK instantly so Render detects the open port immediately
    return {"status": "healthy", "service": "AeroEstate RAG Engine"}


@app.post("/api/rag/query", response_model=QueryResponse)
def handle_rag_query(payload: QueryRequest):
    """
    Endpoint called by n8n when an email is classified as a 'general_query' 
    or 'sales_inquiry' answerable by the Task 5 ChromaDB knowledge base.
    """
    try:
        if not payload.question or not payload.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty.")

        print(f"📩 Processing RAG query from email [{payload.email_sender}]: '{payload.question}'")

        # Lazy load pipeline on first request
        pipeline = get_rag_pipeline()
        result = pipeline.run(payload.question)

        answer_text = result.get("answer", "Information not found in knowledge base.")
        raw_context = result.get("context", [])

        # Process raw context documents into clean string representations
        context_strings: List[str] = []
        if isinstance(raw_context, list):
            for item in raw_context:
                if isinstance(item, str):
                    context_strings.append(item)
                elif isinstance(item, dict):
                    text_content = item.get("text") or item.get("page_content") or item.get("content") or str(item)
                    context_strings.append(str(text_content))
                else:
                    page_content = getattr(item, "page_content", str(item))
                    context_strings.append(str(page_content))
        elif raw_context:
            context_strings = [str(raw_context)]

        return QueryResponse(
            answer=answer_text,
            context=context_strings,
            status="success"
        )

    except Exception as e:
        print(f"❌ Error processing RAG query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"RAG processing failed: {str(e)}")