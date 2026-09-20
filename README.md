# AeroEstate AI Support — Intelligent Email Triage & RAG System

An end-to-end automated email processing pipeline built using **n8n**, **Groq LLMs**, **FastAPI**, **ChromaDB**, **Supabase**, and **Discord**. 

The system automatically classifies incoming customer emails, generates grounded RAG (Retrieval-Augmented Generation) responses for general inquiries, routes job applications to HR, creates drafts for meeting requests, and logs all transactional metadata to Supabase in real time.

**project Link**: https://drive.google.com/drive/folders/1gdm9BWa4miZHM8x3gykxjh9oVbemcEIS?usp=sharing

**LinkedIn Post**: https://lnkd.in/p/dV4PRgqK

---

## 🏗️ Architecture Overview

```text
                                  +-------------------+
                                  |   Gmail Trigger   |
                                  +---------+---------+
                                            |
                                            v
                                  +-------------------+
                                  | Groq LLM Classifier|
                                  +---------+---------+
                                            |
                                            v
                                  +-------------------+
                                  | Clean & Extract   |
                                  +---------+---------+
                                            |
                                            v
                                  +-------------------+
                                  |    Switch Node    |
                                  +----+----+----+----+
                                       |    |    |
             +-------------------------+    |    +-------------------------+
             |                              |                              |
             v                              v                              v
   [RAG Reply Branch]             [Job Application Branch]          [Meeting Request Branch]
             |                              |                              |
             v                              v                              v
+------------------------+      +------------------------+      +------------------------+
| FastAPI RAG Endpoint   |      | Forward to HR (Gmail)  |      | Create Draft (Gmail)   |
| (ChromaDB + LangChain) |      +-----------+------------+      +-----------+------------+
+-----------+------------+                  |                              |
            |                               v                              v
            v                     +------------------------+      +------------------------+
+------------------------+      | Discord (#hr-alerts)   |      | Discord (#manager-alerts)|
| Reply to Message (Gmail|      +-----------+------------+      +-----------+------------+
+-----------+------------+                  |                              |
            |                               |                              |
            +-------------------------------+------------------------------+
                                            |
                                            v
                                 +----------------------+
                                 |  Supabase Logging    |
                                 |  (email_logs table)  |
                                 +----------------------+
```
## ✨ Key Features
AI Classification & Extraction: Uses Groq (openai/gpt-oss-120b) to parse email intent, assigning categories (general_query, job_application, meeting_request, urgent_request) and priority levels (low, normal, high).

Grounded RAG Pipeline: Connects to a custom FastAPI backend powering a ChromaDB knowledge base to generate precise, brand-aligned responses without hallucinations or raw code artifacts.

Automated HR & Manager Routing: Instant notification routing to dedicated Discord channels (#hr-alerts, #manager-alerts, #urgent) with parsed metadata (sender, candidate name, subject, priority).

Smart Threading & Auto-Drafting: Maintains clean email conversation threads using inline Gmail reply headers and creates automated drafts for calendar scheduling.

Centralized Supabase Audit Logging: Records message IDs, category tags, action taken, and execution timestamps into a email_logs table.

## 📂 Repository Structure
.
├── app/
│   ├── config/          # Environment & API settings
│   ├── embeddings/      # Vector embedding provider setups
│   ├── llm/             # LLM provider configurations
│   ├── pipelines/       # RAG pipeline logic (rag_pipeline.py)
│   └── vectorstore/     # ChromaDB store & custom retriever implementation
├── main.py              # FastAPI API router exposing /api/rag/query
├── AeroEstate_Email_Triage_Workflow.json # Exported n8n workflow file
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation


## 🛠️ Environment Variables Setup
Create a .env file in the project root directory:

Code snippet
# LLM & Embedding Credentials
GROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=your_openai_api_key

# Vector DB Settings
CHROMA_PERSIST_DIR=./data/chroma_db

# FastAPI Configuration
HOST=0.0.0.0
PORT=8000

## 🚀 Local Installation & Running
1. Clone & Set Up Backend
Bash
# Clone the repository
git clone [https://github.com/YOUR_USERNAME/aeroestate-email-triage.git](https://github.com/YOUR_USERNAME/aeroestate-email-triage.git)
cd aeroestate-email-triage

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the FastAPI backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
Verify backend health at: http://localhost:8000/

2. Import & Configure Workflow in n8n
Open your n8n instance.

Select Workflows -> Import from File.

Upload AeroEstate_Email_Triage_Workflow.json.

Update node credentials:

Gmail Trigger / Gmail Nodes: Link your authorized Google Workspace account.

HTTP Request Node: Set endpoint URL to your live FastAPI endpoint (/api/rag/query).

Discord Nodes: Set your Discord Webhook URLs for each alert channel.

Supabase Node: Select your Supabase connection and target table email_logs.

## 📊 Supabase Schema (email_logs)
Column Name,Type,Description
id,uuid,Primary Key
gmail_message_id,text,ID of processed Gmail message
sender,text,Sender email address
category,text,Assigned workflow category
priority,text,"low, normal, or high"
action_taken,text,"Action performed (auto_replied, forwarded_hr, draft_created)"
rag_answer,text,AI-generated reply body (or null for non-RAG paths)
processed_at,timestamp,Execution timestamp

## 🧪 Testing & Verification
General Inquiry: Send an email asking "What are your property management pricing options?".

Expected Output: Receives a styled HTML auto-reply from AeroEstate Support detailing fees and services.

Job Application: Send an email with subject "Senior AI Engineer Application - Jane Doe".

Expected Output: Forwards the email to HR and fires a formatted alert card in #hr-alerts.

Meeting Request: Send an email asking "Can we schedule a call next week?".

Expected Output: Creates an auto-draft reply in Gmail and posts a notification to #manager-alerts.
