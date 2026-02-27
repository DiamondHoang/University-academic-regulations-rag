# from fastapi import FastAPI
# from pydantic import BaseModel
# from fastapi.responses import HTMLResponse
# from uni_rag import UniversityRAG
# from loader.doc_loader import RegulationDocumentLoader
# from config import Config

# app = FastAPI()

# # Load system khi start server
# loader = RegulationDocumentLoader(base_path=Config.BASE_PATH)
# rag = UniversityRAG(config={"use_hybrid": True})

# documents = loader.load_documents()
# rag.build_vectorstore(documents, force_rebuild=False)


# class QueryRequest(BaseModel):
#     question: str


# @app.post("/query")
# def query_rag(request: QueryRequest):
#     answer = rag.query(request.question)
#     return {"answer": answer}


# @app.get("/", response_class=HTMLResponse)
# def home():
#     return """
#     <html>
#         <head>
#             <title>University Chatbot</title>
#         </head>
#         <body>
#             <h2>Chatbot Quy Định</h2>
#             <input type="text" id="question" size="60"/>
#             <button onclick="send()">Gửi</button>
#             <pre id="response"></pre>

#             <script>
#                 async function send() {
#                     const question = document.getElementById("question").value;
#                     const response = await fetch("/query", {
#                         method: "POST",
#                         headers: {
#                             "Content-Type": "application/json"
#                         },
#                         body: JSON.stringify({ question })
#                     });
#                     const data = await response.json();
#                     document.getElementById("response").innerText = data.answer;
#                 }
#             </script>
#         </body>
#     </html>
#     """




import uuid
import asyncio
import traceback
from datetime import datetime
from typing import Optional, List, Dict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from config import Config
from loader.doc_loader import RegulationDocumentLoader
from uni_rag import UniversityRAG
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="University Regulation Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thread pool — keeps async event loop free while RAG runs
_executor = ThreadPoolExecutor(max_workers=4)

# In-memory session store
sessions: Dict[str, dict] = {}

# Shared RAG (vectorstore built once, shared across sessions) 
_shared_rag: Optional[UniversityRAG] = None
_rag_error: Optional[str] = None


def _build_shared_rag():
    """Build the shared RAG — runs in a thread on startup."""
    global _shared_rag, _rag_error
    try:
        loader = RegulationDocumentLoader(base_path=Config.BASE_PATH)
        rag = UniversityRAG(config={"use_hybrid": True})
        documents = loader.load_documents()
        rag.build_vectorstore(documents, force_rebuild=False)
        _shared_rag = rag
    except Exception as e:
        _rag_error = str(e)
        print(f"RAG initialization failed: {e}")
        traceback.print_exc()


def _make_session_rag() -> UniversityRAG:
    """New RAG instance sharing the already-built vectorstore/retriever."""
    base = _shared_rag
    session_rag = UniversityRAG(config={"use_hybrid": True})
    session_rag.vectorstore = base.vectorstore
    session_rag.retriever  = base.retriever
    session_rag.all_chunks = base.all_chunks
    return session_rag


# Pydantic models

class NewSessionResponse(BaseModel):
    session_id: str
    title: str
    created_at: str

class ChatRequest(BaseModel):
    session_id: str
    message: str
    doc_type: Optional[str] = None
    regulation_type: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    session_id: str

class SessionSummary(BaseModel):
    session_id: str
    title: str
    created_at: str
    message_count: int
    last_message: Optional[str] = None

class MessageItem(BaseModel):
    role: str
    content: str
    timestamp: str

class SessionDetail(BaseModel):
    session_id: str
    title: str
    created_at: str
    messages: List[MessageItem]


# Lifecycle

@app.on_event("startup")
async def startup_event():
    loop = asyncio.get_event_loop()
    # Non-blocking: RAG init runs in background thread
    loop.run_in_executor(_executor, _build_shared_rag)


# Health / debug 

@app.get("/health")
async def health():
    """Use this to check if RAG finished loading before sending messages."""
    if _rag_error:
        return JSONResponse(status_code=500, content={"status": "error", "detail": _rag_error})
    if _shared_rag is None:
        return JSONResponse(status_code=503, content={"status": "loading", "detail": "RAG is still initializing..."})
    return {"status": "ok", "vectorstore": "ready"}


# Session routes 

@app.post("/sessions", response_model=NewSessionResponse)
async def create_session():
    if _shared_rag is None:
        raise HTTPException(status_code=503, detail="RAG is still initializing. Check /health and retry.")
    session_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    sessions[session_id] = {
        "rag": _make_session_rag(),
        "title": "New conversation",
        "messages": [],
        "created_at": now,
    }
    return NewSessionResponse(session_id=session_id, title="New conversation", created_at=now)


@app.get("/sessions", response_model=List[SessionSummary])
async def list_sessions():
    result = []
    for sid, data in sessions.items():
        msgs = data["messages"]
        last = msgs[-1]["content"][:80] if msgs else None
        result.append(SessionSummary(
            session_id=sid,
            title=data["title"],
            created_at=data["created_at"],
            message_count=len(msgs),
            last_message=last,
        ))
    result.sort(key=lambda s: s.created_at, reverse=True)
    return result


@app.get("/sessions/{session_id}", response_model=SessionDetail)
async def get_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    data = sessions[session_id]
    return SessionDetail(
        session_id=session_id,
        title=data["title"],
        created_at=data["created_at"],
        messages=[MessageItem(**m) for m in data["messages"]],
    )


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    del sessions[session_id]
    return {"status": "deleted"}


@app.delete("/sessions/{session_id}/history")
async def clear_history(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    sessions[session_id]["messages"] = []
    sessions[session_id]["rag"].memory.clear()
    return {"status": "cleared"}


@app.patch("/sessions/{session_id}/title")
async def rename_session(session_id: str, body: dict):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    sessions[session_id]["title"] = body.get("title", "Untitled")
    return {"status": "ok"}


# Chat route 

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if _shared_rag is None:
        raise HTTPException(status_code=503, detail="RAG is still initializing. Check /health and retry in a moment.")
    if req.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found. Create a session first.")

    data = sessions[req.session_id]
    rag: UniversityRAG = data["rag"]
    now = datetime.utcnow().isoformat()

    data["messages"].append({"role": "user", "content": req.message, "timestamp": now})

    if len(data["messages"]) == 1:
        data["title"] = req.message[:60] + ("…" if len(req.message) > 60 else "")

    def _run_query():
        return rag.query(
            question=req.message,
            doc_type=req.doc_type,
            regulation_type=req.regulation_type,
        )

    try:
        loop = asyncio.get_event_loop()
        answer = await loop.run_in_executor(_executor, _run_query)
    except Exception as e:
        traceback.print_exc()
        answer = f"Xin lỗi, có lỗi xảy ra khi xử lý câu hỏi: {str(e)}"

    data["messages"].append({
        "role": "assistant",
        "content": answer,
        "timestamp": datetime.utcnow().isoformat(),
    })

    return ChatResponse(answer=answer, session_id=req.session_id)


# Serve frontend

FRONTEND = Path(__file__).parent / "index.html"

@app.get("/")
async def serve_frontend():
    if FRONTEND.exists():
        return FileResponse(FRONTEND)
    return {"message": "Place index.html in the same folder as server.py"}