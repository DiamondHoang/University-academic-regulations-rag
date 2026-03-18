import uuid
import asyncio
import traceback
from datetime import datetime
from typing import Optional, List, Dict
from pathlib import Path
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from config import Config
from loader.doc_loader import RegulationDocumentLoader
from uni_rag import UniversityRAG
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for FastAPI."""
    # Run RAG init as a background task in the main event loop
    asyncio.create_task(_build_shared_rag())
    yield
    # Shutdown
    _executor.shutdown(wait=False)

app = FastAPI(title="University Regulation Chatbot", lifespan=lifespan)

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


async def _build_shared_rag():
    """Build the shared RAG — runs as an async task on startup."""
    global _shared_rag, _rag_error
    try:
        # 1. Disk/CPU intensive loading in a thread
        loader = RegulationDocumentLoader(base_path=Config.BASE_PATH)
        documents = await asyncio.to_thread(loader.load_documents)
        
        # 2. Create RAG instance in the MAIN LOOP (ensures async components capture correct loop)
        rag = UniversityRAG()
        
        # 3. Vectorstore building in a thread
        await asyncio.to_thread(rag.build_vectorstore, documents, force_rebuild=False)
        
        _shared_rag = rag
        print("RAG initialization complete.")
    except Exception as e:
        _rag_error = str(e)
        print(f"RAG initialization failed: {e}")
        traceback.print_exc()


def _make_session_rag() -> UniversityRAG:
    """New RAG instance sharing the already-built vectorstore/retriever."""
    if _shared_rag is None:
         # Fallback just in case, though it should be initialized by now
         _build_shared_rag()
         
    base = _shared_rag
    session_rag = UniversityRAG()
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




# Health / debug 

@app.get("/health")
async def health():
    """Use this to check if RAG finished loading before sending messages."""
    if _rag_error:
        return JSONResponse(status_code=500, content={"status": "error", "detail": _rag_error})
    if _shared_rag is None:
        # Return 200 instead of 503 so Azure warmup probes don't fail during initialization
        return JSONResponse(status_code=200, content={"status": "loading", "detail": "RAG is still initializing..."})
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

@app.post("/chat")
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

    async def _streamer():
        full_answer = ""
        try:
            async for chunk in rag.astream_query(
                question=req.message,
                doc_type=req.doc_type,
                regulation_type=req.regulation_type,
            ):
                full_answer += chunk
                yield chunk
            
            # Record the full answer in session history after stream ends
            data["messages"].append({
                "role": "assistant",
                "content": full_answer,
                "timestamp": datetime.utcnow().isoformat(),
            })
        except Exception as e:
            traceback.print_exc()
            err_msg = f"\n[Lỗi: {str(e)}]"
            yield err_msg

    return StreamingResponse(_streamer(), media_type="text/plain")


# Serve frontend

FRONTEND = Path(__file__).parent / "index.html"

@app.get("/")
async def serve_frontend():
    if FRONTEND.exists():
        return FileResponse(FRONTEND)
    return {"message": "Place index.html in the same folder as server.py"}

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
