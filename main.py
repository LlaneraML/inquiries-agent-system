import os
import time
import io
import re
import hashlib
from datetime import datetime
from enum import Enum
from typing import Optional, List

from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

import chromadb
from google import genai
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
import pymysql

load_dotenv()

# =========================================================
# CONFIGURATION & ENVIRONMENT VARIABLES
# =========================================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing from .env file.")

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "inquiries_agent_db")
DB_PORT = int(os.getenv("DB_PORT", 3306))

def get_db_connection():
    try:
        return pymysql.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            port=DB_PORT,
            cursorclass=pymysql.cursors.DictCursor,
            autocommit=True,
            ssl={"ssl_mode": "VERIFY_IDENTITY"}
        )
    except Exception as e:
        print(f"⚠️ TiDB Connection Error: {e}")
        return None

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

# =========================================================
# DATABASE TABLE INITIALIZATION
# =========================================================
def init_mysql_tables():
    conn = get_db_connection()
    if conn:
        with conn.cursor() as cursor:
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS offices (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(100) NOT NULL UNIQUE,
                code VARCHAR(20) NOT NULL UNIQUE,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(50) UNIQUE NOT NULL,
                password VARCHAR(255) NOT NULL,
                full_name VARCHAR(100) NOT NULL,
                role VARCHAR(20) DEFAULT 'student',
                office_id INT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (office_id) REFERENCES offices(id) ON DELETE SET NULL
            );
            """)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INT AUTO_INCREMENT PRIMARY KEY,
                session_id VARCHAR(100) NOT NULL,
                role VARCHAR(20) NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX (session_id)
            );
            """)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS unanswered_logs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                session_id VARCHAR(100) NOT NULL,
                user_message TEXT NOT NULL,
                office_id INT NULL,
                status ENUM('pending', 'resolved') DEFAULT 'pending',
                office_reply TEXT NULL,
                resolved_at TIMESTAMP NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (office_id) REFERENCES offices(id) ON DELETE SET NULL
            );
            """)
        conn.close()

init_mysql_tables()

# =========================================================
# FASTAPI APP & AUTHENTICATION SETUP
# =========================================================
app = FastAPI(title="Inquiries Agent API", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserRole(str, Enum):
    ADMIN = "admin"
    STUDENT = "student"
    EMPLOYEE = "employee"

def get_current_user():
    # Placeholder user context (returns admin role for testing)
    return {"username": "admin", "role": "admin"}

def require_roles(allowed_roles: List[UserRole]):
    def role_checker(current_user: dict = Depends(get_current_user)):
        if current_user.get("role") not in [r.value for r in allowed_roles]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied for your role."
            )
        return current_user
    return role_checker

# =========================================================
# AI CLIENT & VECTOR DB SETUP
# =========================================================
ai_client = genai.Client(api_key=GEMINI_API_KEY)
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="university_knowledge_base")

MAX_HISTORY_TURNS = 6

# =========================================================
# PYDANTIC REQUEST & RESPONSE SCHEMAS
# =========================================================
class RegisterRequest(BaseModel):
    username: str
    password: str
    full_name: str

class LoginRequest(BaseModel):
    username: str
    password: str

class ChatRequest(BaseModel):
    message: str
    session_id: str
    office_id: Optional[int] = None

class ChatResponse(BaseModel):
    reply: str
    source: str
    session_id: str
    department_forward: Optional[str] = None
    response_time_seconds: float

class ResolveLogRequest(BaseModel):
    log_id: int
    reply: str

HARDCODED_FAQS = {
    "hello": "Hello! I am the Inquiries Agent. How can I assist you today?",
    "hi": "Hi there! Ask me anything about university policies, campus events, faculty, or administrative contacts.",
    "contact": "General Directory | Phone: (033) 123-4567 | Email: info@university.edu.ph",
    "portal": "You can access the student portal at: https://portal.university.edu.ph"
}

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def save_chat_turn(session_id: str, role: str, content: str):
    conn = get_db_connection()
    if conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "INSERT INTO chat_history (session_id, role, content) VALUES (%s, %s, %s)",
                (session_id, role, content)
            )
        conn.close()

def get_session_history_from_db(session_id: str) -> list[dict[str, str]]:
    conn = get_db_connection()
    if not conn:
        return []
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT role, content FROM chat_history WHERE session_id = %s ORDER BY id DESC LIMIT %s",
            (session_id, MAX_HISTORY_TURNS)
        )
        rows = cursor.fetchall()
    conn.close()
    return list(reversed(rows))

def log_unanswered_query_to_db(session_id: str, user_message: str, office_id: Optional[int] = None):
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO unanswered_logs (session_id, user_message, office_id, status)
            VALUES (%s, %s, %s, 'pending')
            """,
            (session_id, user_message, office_id)
        )
        conn.commit()
        cursor.close()
        conn.close()

def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def reciprocal_rank_fusion(vector_docs: list[str], bm25_docs: list[str], k: int = 60, top_n: int = 3) -> list[str]:
    scores = {}
    for rank, doc in enumerate(vector_docs):
        scores[doc] = scores.get(doc, 0.0) + (1.0 / (k + rank + 1))
    for rank, doc in enumerate(bm25_docs):
        scores[doc] = scores.get(doc, 0.0) + (1.0 / (k + rank + 1))
    sorted_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [doc for doc, score in sorted_docs[:top_n]]

# =========================================================
# API ENDPOINTS
# =========================================================
@app.get("/")
def root():
    return {"status": "Online", "message": "Inquiries Agent System API is operational."}

@app.post("/auth/register")
def register_user(req: RegisterRequest):
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed.")
    hashed_pwd = hash_password(req.password)
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "INSERT INTO users (username, password, full_name, role) VALUES (%s, %s, %s, %s)",
                (req.username.strip(), hashed_pwd, req.full_name.strip(), "student")
            )
        conn.close()
        return {"status": "Success", "message": "Student account registered successfully!"}
    except pymysql.err.IntegrityError:
        conn.close()
        raise HTTPException(status_code=400, detail="Username already exists.")

@app.post("/auth/login")
def login_user(req: LoginRequest):
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed.")
    hashed_pwd = hash_password(req.password)
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT id, username, full_name, role FROM users WHERE username = %s AND password = %s",
            (req.username.strip(), hashed_pwd)
        )
        user = cursor.fetchone()
    conn.close()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    return {
        "status": "Success",
        "user": user,
        "session_id": f"student_{user['username']}"
    }

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    start_time = time.time()
    user_query = request.message.strip()
    session_id = request.session_id.strip()
    query_lower = user_query.lower()

    # LAYER 1: Rule-Based FAQ
    for key, value in HARDCODED_FAQS.items():
        if key in query_lower:
            save_chat_turn(session_id, "user", user_query)
            save_chat_turn(session_id, "assistant", value)
            return ChatResponse(
                reply=value, 
                source="rule_based", 
                session_id=session_id,
                response_time_seconds=round(time.time() - start_time, 4)
            )

    # LAYER 2: Hybrid Retrieval (BM25 + ChromaDB Vector)
    db_data = collection.get()
    all_docs = db_data.get("documents", []) if db_data else []

    retrieved_docs = []
    if all_docs:
        try:
            emb_res = ai_client.models.embed_content(
                model="gemini-embedding-001",
                contents=user_query,
            )
            query_vector = emb_res.embeddings[0].values
            v_results = collection.query(query_embeddings=[query_vector], n_results=5)
            vector_docs = v_results["documents"][0] if v_results["documents"] else []
        except Exception:
            vector_docs = []

        tokenized_corpus = [tokenize(doc) for doc in all_docs]
        bm25 = BM25Okapi(tokenized_corpus)
        query_tokens = tokenize(user_query)
        bm25_docs = bm25.get_top_n(query_tokens, all_docs, n=5)

        retrieved_docs = reciprocal_rank_fusion(vector_docs, bm25_docs, top_n=3)

    if not retrieved_docs:
        log_unanswered_query_to_db(session_id, user_query, request.office_id)
        fallback_msg = "No official university record found for this topic. Your inquiry has been routed to the department queue."
        save_chat_turn(session_id, "user", user_query)
        save_chat_turn(session_id, "assistant", fallback_msg)
        return ChatResponse(
            reply=fallback_msg,
            source="fallback_router",
            session_id=session_id,
            response_time_seconds=round(time.time() - start_time, 4)
        )

    context_text = "\n---\n".join(retrieved_docs)
    history_logs = get_session_history_from_db(session_id)
    history_str = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history_logs]) if history_logs else "No prior interaction."

    prompt = f"""
    You are an official University Information Assistant for the Inquiries Agent System.
    Answer the student's question accurately based ONLY on the provided public university context excerpts and the prior conversation history.

    Guidelines:
    1. Resolve pronouns and ambiguous references using the Recent Conversation History.
    2. You may count, summarize, or list items directly present in the context excerpts.
    3. If the context and history do not contain relevant information to answer the question, reply strictly with: "UNANSWERED".

    University Context Excerpts:
    {context_text}

    Recent Conversation History:
    {history_str}

    Student Current Question: {user_query}
    """

    try:
        response = ai_client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt,
        )
        answer = response.text.strip()

        if "UNANSWERED" in answer.upper():
            log_unanswered_query_to_db(session_id, user_query, request.office_id)
            fallback_msg = "This detail is not explicitly found in our public university records. Your request was forwarded to the department queue."
            save_chat_turn(session_id, "user", user_query)
            save_chat_turn(session_id, "assistant", fallback_msg)
            return ChatResponse(
                reply=fallback_msg,
                source="fallback_router",
                session_id=session_id,
                response_time_seconds=round(time.time() - start_time, 4)
            )

        save_chat_turn(session_id, "user", user_query)
        save_chat_turn(session_id, "assistant", answer)

        return ChatResponse(
            reply=answer, 
            source="rag_hybrid_gemini", 
            session_id=session_id,
            response_time_seconds=round(time.time() - start_time, 4)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API Error: {str(e)}")

# =========================================================
# OFFICE MANAGEMENT ENDPOINTS
# =========================================================
@app.post("/offices", dependencies=[Depends(require_roles([UserRole.ADMIN]))])
def create_office(name: str, code: str, description: Optional[str] = None):
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO offices (name, code, description) VALUES (%s, %s, %s)",
        (name, code, description)
    )
    conn.commit()
    cursor.close()
    conn.close()
    return {"message": f"Office '{name}' created successfully."}

@app.get("/offices")
def list_offices():
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, code, description FROM offices")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    
    offices = []
    for row in rows:
        if isinstance(row, dict):
            offices.append(row)
        else:
            offices.append({
                "id": row[0],
                "name": row[1],
                "code": row[2],
                "description": row[3]
            })

    return {"offices": offices}

# =========================================================
# UNANSWERED LOG RESOLUTION ENDPOINTS
# =========================================================
@app.get("/admin/unanswered")
def get_office_unanswered_logs(current_user: dict = Depends(require_roles([UserRole.ADMIN, UserRole.EMPLOYEE]))):
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    cursor = conn.cursor()
    
    if current_user.get("role") == "employee" and current_user.get("office_id"):
        cursor.execute(
            "SELECT id, session_id, user_message, office_id, status, created_at FROM unanswered_logs WHERE office_id = %s AND status = 'pending' ORDER BY created_at DESC", 
            (current_user["office_id"],)
        )
    else:
        cursor.execute("SELECT id, session_id, user_message, office_id, status, created_at FROM unanswered_logs WHERE status = 'pending' ORDER BY created_at DESC")
        
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    logs = []
    for row in rows:
        if isinstance(row, dict):
            logs.append(row)
        else:
            logs.append({
                "id": row[0],
                "session_id": row[1],
                "user_message": row[2],
                "office_id": row[3],
                "status": row[4],
                "created_at": str(row[5]) if len(row) > 5 else None
            })

    return {"unanswered_logs": logs}

@app.post("/admin/resolve-log", dependencies=[Depends(require_roles([UserRole.ADMIN, UserRole.EMPLOYEE]))])
def resolve_unanswered_log(payload: ResolveLogRequest):
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    cursor = conn.cursor()
    # 1. Fetch user_message for embedding
    cursor.execute("SELECT user_message FROM unanswered_logs WHERE id = %s", (payload.log_id,))
    row = cursor.fetchone()
    if not row:
        cursor.close()
        conn.close()
        raise HTTPException(status_code=404, detail="Log entry not found")
    
    user_message = row['user_message'] if isinstance(row, dict) else row[0]

    # 2. Update status in MySQL
    cursor.execute(
        """
        UPDATE unanswered_logs 
        SET status = 'resolved', office_reply = %s, resolved_at = NOW() 
        WHERE id = %s
        """,
        (payload.reply, payload.log_id)
    )
    conn.commit()
    cursor.close()
    conn.close()

    # 3. Automatically index Q&A into ChromaDB so RAG learns the answer
    try:
        qa_doc = f"Question: {user_message}\nAnswer: {payload.reply}"
        emb_res = ai_client.models.embed_content(
            model="gemini-embedding-001",
            contents=qa_doc,
        )
        embedding = list(emb_res.embeddings[0].values)
        collection.upsert(
            documents=[qa_doc],
            embeddings=[embedding],  # type: ignore
            metadatas=[{"source": "resolved_inquiry", "log_id": payload.log_id}],
            ids=[f"resolved_log_{payload.log_id}"]
        )
    except Exception as e:
        print(f"⚠️ ChromaDB indexing warning: {e}")

    return {"message": "Inquiry resolved and added to knowledge base successfully."}

@app.post("/admin/sync-resolved-to-chroma", dependencies=[Depends(require_roles([UserRole.ADMIN]))])
def sync_resolved_to_chroma():
    """One-time sync to push all previously resolved MySQL logs into ChromaDB."""
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    cursor = conn.cursor()
    cursor.execute("SELECT id, user_message, office_reply FROM unanswered_logs WHERE status = 'resolved'")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    count = 0
    for row in rows:
        log_id = row['id'] if isinstance(row, dict) else row[0]
        user_msg = row['user_message'] if isinstance(row, dict) else row[1]
        reply = row['office_reply'] if isinstance(row, dict) else row[2]

        if user_msg and reply:
            qa_doc = f"Question: {user_msg}\nAnswer: {reply}"
            emb_res = ai_client.models.embed_content(
                model="gemini-embedding-001",
                contents=qa_doc,
            )
            embedding = list(emb_res.embeddings[0].values)
            collection.upsert(
                documents=[qa_doc],
                embeddings=[embedding],  # type: ignore
                metadatas=[{"source": "resolved_inquiry", "log_id": log_id}],
                ids=[f"resolved_log_{log_id}"]
            )
            count += 1

    return {"message": f"Successfully synced {count} resolved inquiries into ChromaDB vector store."}

# =========================================================
# PDF HANDBOOK UPLOAD ENDPOINT
# =========================================================
@app.post("/admin/upload-pdf")
async def upload_pdf_handbook(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    start_time = time.time()
    contents = await file.read()
    pdf_reader = PdfReader(io.BytesIO(contents))
    
    os.makedirs("documents", exist_ok=True)
    with open(os.path.join("documents", file.filename), "wb") as f:
        f.write(contents)

    all_chunks, all_metadatas, all_ids = [], [], []
    chunk_counter = collection.count()

    for page_num, page in enumerate(pdf_reader.pages):
        text = page.extract_text()
        if text and text.strip():
            chunks = chunk_text(text)
            for chunk in chunks:
                chunk_counter += 1
                all_chunks.append(chunk)
                all_metadatas.append({"source": file.filename, "page": page_num + 1})
                all_ids.append(f"upload_{chunk_counter}")

    if not all_chunks:
        raise HTTPException(status_code=400, detail="Could not extract text from PDF.")

    batch_size = 20
    for i in range(0, len(all_chunks), batch_size):
        batch_chunks = all_chunks[i:i + batch_size]
        batch_metadatas = all_metadatas[i:i + batch_size]
        batch_ids = all_ids[i:i + batch_size]

        emb_response = ai_client.models.embed_content(
            model="gemini-embedding-001",
            contents=batch_chunks,
        )
        
        batch_embeddings = [list(emb.values) for emb in emb_response.embeddings]  # type: ignore
        
        collection.upsert(
            documents=batch_chunks,
            embeddings=batch_embeddings,  # type: ignore
            metadatas=batch_metadatas,
            ids=batch_ids
        )

    return {
        "status": "Success",
        "filename": file.filename,
        "total_pages": len(pdf_reader.pages),
        "total_chunks_indexed": len(all_chunks),
        "indexing_time_seconds": round(time.time() - start_time, 2),
        "message": f"Indexed '{file.filename}' successfully."
    }