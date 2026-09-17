import os
import time
import io
import re
import hashlib
from datetime import datetime
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import chromadb
from google import genai
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
import pymysql

load_dotenv()

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
            autocommit=True
        )
    except Exception as e:
        print(f"⚠️ MySQL Connection Error: {e}")
        return None

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def init_mysql_tables():
    conn = get_db_connection()
    if conn:
        with conn.cursor() as cursor:
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(50) UNIQUE NOT NULL,
                password VARCHAR(255) NOT NULL,
                full_name VARCHAR(100) NOT NULL,
                role VARCHAR(20) DEFAULT 'student',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                session_id VARCHAR(100) NOT NULL,
                user_query TEXT NOT NULL,
                routed_office VARCHAR(255) NOT NULL
            );
            """)
        conn.close()

init_mysql_tables()

app = FastAPI(title="Inquiries Agent API", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ai_client = genai.Client(api_key=GEMINI_API_KEY)
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="university_knowledge_base")

MAX_HISTORY_TURNS = 6

class RegisterRequest(BaseModel):
    username: str
    password: str
    full_name: str

class LoginRequest(BaseModel):
    username: str
    password: str

class ChatRequest(BaseModel):
    message: str
    session_id: str = "guest_session"

class ChatResponse(BaseModel):
    reply: str
    source: str
    session_id: str
    department_forward: str | None = None
    response_time_seconds: float

HARDCODED_FAQS = {
    "hello": "Hello! I am the Inquiries Agent. How can I assist you today?",
    "hi": "Hi there! Ask me anything about university policies, campus events, faculty, or administrative contacts.",
    "contact": "General Directory | Phone: (033) 123-4567 | Email: info@university.edu.ph",
    "portal": "You can access the student portal at: https://portal.university.edu.ph"
}

OFFICE_DIRECTORY = {
    "enrollment": "Registrar's Office (registrar@university.edu.ph)",
    "tuition": "Accounting & Finance Office (finance@university.edu.ph)",
    "fee": "Accounting & Finance Office (finance@university.edu.ph)",
    "dorm": "Student Affairs Office (sao@university.edu.ph)",
    "club": "Student Affairs Office (sao@university.edu.ph)",
    "professor": "College Dean's Office / Academic Affairs (academic.affairs@university.edu.ph)",
    "faculty": "College Dean's Office / Academic Affairs (academic.affairs@university.edu.ph)",
    "event": "Public Relations & Campus Events Office (events@university.edu.ph)",
    "default": "General Helpdesk (helpdesk@university.edu.ph)"
}

def detect_office_routing(query: str) -> str:
    query_lower = query.lower()
    for keyword, office in OFFICE_DIRECTORY.items():
        if keyword in query_lower:
            return office
    return OFFICE_DIRECTORY["default"]

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

def log_unanswered_query_to_db(session_id: str, query: str, routed_office: str):
    conn = get_db_connection()
    if conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "INSERT INTO unanswered_logs (timestamp, session_id, user_query, routed_office) VALUES (%s, %s, %s, %s)",
                (datetime.now(), session_id, query, routed_office)
            )
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
        target_office = detect_office_routing(user_query)
        log_unanswered_query_to_db(session_id, user_query, target_office)
        fallback_msg = f"No official university record found for this topic. Your inquiry has been routed to: {target_office}."
        save_chat_turn(session_id, "user", user_query)
        save_chat_turn(session_id, "assistant", fallback_msg)
        return ChatResponse(
            reply=fallback_msg,
            source="fallback_router",
            session_id=session_id,
            department_forward=target_office,
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
            target_office = detect_office_routing(user_query)
            log_unanswered_query_to_db(session_id, user_query, target_office)
            fallback_msg = f"This detail is not explicitly found in our public university records. Your request was forwarded to: {target_office}."
            save_chat_turn(session_id, "user", user_query)
            save_chat_turn(session_id, "assistant", fallback_msg)
            return ChatResponse(
                reply=fallback_msg,
                source="fallback_router",
                session_id=session_id,
                department_forward=target_office,
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

@app.get("/admin/unanswered-logs")
def get_unanswered_logs():
    conn = get_db_connection()
    if not conn:
        return {"total_unanswered": 0, "logs": []}
    with conn.cursor() as cursor:
        cursor.execute("SELECT timestamp as Timestamp, session_id as Session_ID, user_query as User_Query, routed_office as Routed_Office FROM unanswered_logs ORDER BY id DESC")
        logs = cursor.fetchall()
    conn.close()

    for log in logs:
        if isinstance(log["Timestamp"], datetime):
            log["Timestamp"] = log["Timestamp"].strftime("%Y-%m-%d %H:%M:%S")

    return {"total_unanswered": len(logs), "logs": logs}

@app.post("/admin/upload-pdf")
async def upload_pdf_handbook(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
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
        embeddings = [emb.values for emb in emb_response.embeddings]

        collection.upsert(
            documents=batch_chunks,
            embeddings=embeddings,
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