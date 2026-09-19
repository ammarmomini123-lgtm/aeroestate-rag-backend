import os
import time
from datetime import datetime
import pandas as pd
import streamlit as st

from app.pipelines.ingestion_pipeline import IngestionPipeline
from app.pipelines.rag_pipeline import RAGPipeline
from app.vectorstore.chroma_store import ChromaVectorStore
from app.config.settings import settings

# Page Setup & Configuration
st.set_page_config(
    page_title="AI-Powered Document RAG Chatbot Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional Dashboard Look
st.markdown("""
<style>
    /* Metric Card Styling */
    .metric-card {
        background-color: #1E293B;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #334155;
        text-align: center;
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #38BDF8;
    }
    .metric-label {
        font-size: 14px;
        color: #94A3B8;
    }

    /* WhatsApp Style Chat Layout */
    /* Target USER chat messages -> Move to Right */
    div[data-testid="stChatMessage"]:has(div[aria-label="Chat message from user"]) {
        flex-direction: row-reverse;
        text-align: right;
        background-color: #054C44 !important; /* WhatsApp Dark Green Bubble */
        color: #FFFFFF !important;
        border-radius: 15px 15px 0px 15px !important;
        margin-left: auto !important;
        max-width: 75% !important;
        padding: 12px 16px !important;
    }

    /* Target ASSISTANT chat messages -> Move to Left */
    div[data-testid="stChatMessage"]:has(div[aria-label="Chat message from assistant"]) {
        flex-direction: row;
        text-align: left;
        background-color: #202C33 !important; /* WhatsApp Dark Assistant Bubble */
        color: #E9EDEF !important;
        border-radius: 15px 15px 15px 0px !important;
        margin-right: auto !important;
        max-width: 80% !important;
        padding: 12px 16px !important;
    }

    /* Timestamp alignment inside user bubble */
    div[data-testid="stChatMessage"]:has(div[aria-label="Chat message from user"]) .caption {
        text-align: right;
        color: #8696A0 !important;
    }
</style>
""", unsafe_allow_html=True)


# Cache Heavy Pipelines
@st.cache_resource
def get_vector_store():
    return ChromaVectorStore()

@st.cache_resource
def get_rag_pipeline():
    return RAGPipeline()


# Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "query_count" not in st.session_state:
    st.session_state.query_count = 0


# Helper: Render Sources Cleanly
def render_sources(sources):
    if sources:
        with st.expander("📄 View Grounded Source References", expanded=False):
            for idx, src in enumerate(sources, 1):
                file_name = src.get('file_name', 'Unknown File')
                page_num = src.get('page', 1)
                score = src.get('score', 0.0)
                snippet = src.get('content_snippet', '')
                
                st.markdown(f"**Source {idx}:** `📄 {file_name}` — **Page {page_num}** (Relevance Score: `{score:.4f}`)")
                st.caption(f"Snippet: _{snippet}_")
                if idx < len(sources):
                    st.divider()


# Main Header
st.title("🤖 AI-Powered Document RAG Dashboard")
st.caption("Practical Evaluation Project — Practical Document Q&A & Vector Indexing")

# Sidebar - Quick Actions
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/real-estate.png", width=64)
    st.title("AeroEstate Realty")
    st.markdown("---")
    st.subheader("System Actions")
    
    if st.button("🔄 Refresh Dashboard Data", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.caption("Built with FastAPI, Streamlit, ChromaDB & Gemini 1.5")


# Tabs Navigation
tab1, tab2, tab3 = st.tabs(["📊 Home Dashboard", "📄 Document Management", "💬 Chatbot Interface"])


# ==========================================
# TAB 1: HOME DASHBOARD METRICS
# ==========================================
with tab1:
    st.subheader("System Performance & Knowledge Base Overview")
    
    vector_store = get_vector_store()
    stats = vector_store.get_stats()
    
    total_docs = stats["total_documents"]
    total_chunks = stats["total_indexed_chunks"]
    indexed_files = stats["indexed_files"]
    
    # Calculate file system count
    raw_files = []
    if os.path.exists(settings.RAW_DATA_DIR):
        raw_files = [f for f in os.listdir(settings.RAW_DATA_DIR) if not f.startswith(".")]

    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(raw_files)}</div>
            <div class="metric-label">Total Uploaded Files</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_docs}</div>
            <div class="metric-label">Processed & Indexed Docs</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_chunks}</div>
            <div class="metric-label">Total Indexed Chunks</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{st.session_state.query_count}</div>
            <div class="metric-label">Total Queries Executed</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Current Active Knowledge Base Files")
    if indexed_files:
        df = pd.DataFrame({
            "Document Name": indexed_files,
            "Format": [f.split('.')[-1].upper() for f in indexed_files],
            "Status": ["Indexed & Ready"] * len(indexed_files)
        })
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No documents currently indexed. Go to 'Document Management' tab to upload sample PDFs.")


# ==========================================
# TAB 2: DOCUMENT MANAGEMENT
# ==========================================
with tab2:
    st.subheader("Upload & Manage Knowledge Base Documents")
    
    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, or TXT Files", 
        type=["pdf", "docx", "txt", "md"], 
        accept_multiple_files=True
    )
    
    if st.button("🚀 Process & Ingest Files", type="primary"):
        if uploaded_files:
            os.makedirs(settings.RAW_DATA_DIR, exist_ok=True)
            saved_count = 0
            
            for uploaded_file in uploaded_files:
                save_path = os.path.join(settings.RAW_DATA_DIR, uploaded_file.name)
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                saved_count += 1
            
            with st.spinner(f"Ingesting {saved_count} file(s) into ChromaDB..."):
                ingestion = IngestionPipeline()
                success = ingestion.run(settings.RAW_DATA_DIR)
                
            if success:
                st.success("✓ Document ingestion completed successfully!")
                st.cache_resource.clear()
                time.sleep(1)
                st.rerun()
            else:
                st.error("⚠️ Ingestion failed or no readable content found.")
        else:
            st.warning("Please select at least one document file to upload.")

    st.markdown("---")
    st.subheader("Manage Existing Index")
    
    if indexed_files:
        selected_doc = st.selectbox("Select document to delete:", indexed_files)
        if st.button("🗑️ Delete Selected Document", type="secondary"):
            vector_store.delete_by_file_name(selected_doc)
            
            # Remove file from raw folder
            file_path = os.path.join(settings.RAW_DATA_DIR, selected_doc)
            if os.path.exists(file_path):
                os.remove(file_path)
                
            st.success(f"✓ Deleted '{selected_doc}' from vector index and storage.")
            st.cache_resource.clear()
            time.sleep(1)
            st.rerun()
    else:
        st.write("No documents available to manage.")


# ==========================================
# TAB 3: CHATBOT INTERFACE
# ==========================================
with tab3:
    st.subheader("💬 Interactive Document Q&A Assistant")

    # Render Historical Messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("timestamp"):
                st.caption(f"🕒 {msg['timestamp']}")
            if msg.get("sources"):
                render_sources(msg["sources"])

    # User Input Field
    if user_query := st.chat_input("Ask a question about real estate listings, fees, policies, or FAQs..."):
        now_str = datetime.now().strftime("%H:%M:%S")
        
        # 1. Add User Message to History
        st.session_state.messages.append({
            "role": "user", 
            "content": user_query,
            "timestamp": now_str
        })
        
        # Rerun to immediately render WhatsApp right-aligned user bubble
        st.rerun()

# Assistant Response Generation Logic (Outside input block to preserve clean re-render)
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    last_user_query = st.session_state.messages[-1]["content"]
    
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base & generating answer..."):
            rag = get_rag_pipeline()
            result = rag.run(last_user_query)
            st.session_state.query_count += 1

        answer_text = result.get("answer", "No response generated.")
        sources = result.get("context", [])

        st.markdown(answer_text)
        render_sources(sources)

        # Save Assistant Response to History
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer_text,
            "sources": sources,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
        st.rerun()