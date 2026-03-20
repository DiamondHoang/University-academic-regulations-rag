import uuid
import asyncio
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from config import Config
from loader.doc_loader import RegulationDocumentLoader
from retrieval.response_generator import ResponseGenerator
from langchain_huggingface import HuggingFaceEmbeddings
from uni_rag import UniversityRAG

# ─────────────────────────────────────────────────────────────────────────────
# Session & Resource Management

class SessionManager:
    """Manages RAG sessions, shared resources, and lifecycle."""
    
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.shared_rag: Optional[UniversityRAG] = None
        self.shared_embeddings: Optional[HuggingFaceEmbeddings] = None
        self.shared_generator: Optional[ResponseGenerator] = None
        self.init_error: Optional[str] = None
        self.is_initializing: bool = False

    async def initialize_shared_resources(self):
        """Build the shared RAG and components once."""
        if self.is_initializing:
            return
            
        self.is_initializing = True
        try:
            config = Config.as_dict()
            
            # 1. Load documents and components
            print(f"Loading documents from {Config.BASE_PATH}...")
            loader = RegulationDocumentLoader(base_path=Config.BASE_PATH)
            documents = await asyncio.to_thread(loader.load_documents)
            print(f"Loaded {len(documents)} documents.")
            
            print(f"Initializing embeddings ({config['embedding_model']})...")
            self.shared_embeddings = HuggingFaceEmbeddings(model_name=config["embedding_model"])
            print("Embeddings initialized.")
            
            print("Initializing response generator...")
            self.shared_generator = ResponseGenerator(config)
            print("Response generator initialized.")

            # 2. Build Vectorstore (Disk/CPU intensive)
            print("Initializing RAG core...")
            rag = UniversityRAG(
                embeddings=self.shared_embeddings, 
                response_generator=self.shared_generator
            )
            print(f"Loading/Building vectorstore from {config['db_path']}...")
            await asyncio.to_thread(rag.build_vectorstore, documents, force_rebuild=False)
            print("Vectorstore ready.")
            
            self.shared_rag = rag
            print("RAG shared resources initialization complete.")
        except Exception as e:
            self.init_error = str(e)
        finally:
            self.is_initializing = False

    def create_session(self) -> str:
        """Create a new session with isolated memory but shared retrieval logic."""
        if not self.shared_rag:
            raise RuntimeError("RAG system is not ready.")
            
        session_id = str(uuid.uuid4())
        
        # Create a session-specific RAG instance (shares vectorstore/retriever)
        session_rag = UniversityRAG(
            session_id=session_id,
            embeddings=self.shared_embeddings,
            response_generator=self.shared_generator
        )
        # Link to shared heavy components
        session_rag.vectorstore = self.shared_rag.vectorstore
        session_rag.retriever = self.shared_rag.retriever
        
        self.sessions[session_id] = {
            "rag": session_rag,
            "title": "Cuộc trò chuyện mới",
            "messages": [],
            "created_at": datetime.utcnow().isoformat(),
        }
        return session_id

    def get_session(self, session_id: str) -> Dict[str, Any]:
        if session_id not in self.sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        return self.sessions[session_id]

    def delete_session(self, session_id: str):
        if session_id in self.sessions:
            # Clean up memory persistence
            rag = self.sessions[session_id]["rag"]
            try:
                if hasattr(rag.memory, 'persist_file') and rag.memory.persist_file.exists():
                    rag.memory.persist_file.unlink()
            except Exception:
                pass
            
            del self.sessions[session_id]

    def shutdown(self):
        pass

# Global Manager Instance
manager = SessionManager()

# ─────────────────────────────────────────────────────────────────────────────
# FastAPI Setup

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize RAG and wait until finish to ensure Azure Healthcheck/Traffic is correct
    print("Lifespan: Initializing RAG shared resources...")
    await manager.initialize_shared_resources()
    print("Lifespan: RAG shared resources initialized.")
    yield
    manager.shutdown()

app = FastAPI(title="University Regulation Chatbot", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Pydantic Models

class SessionResponse(BaseModel):
    session_id: str
    title: str
    created_at: str

class ChatRequest(BaseModel):
    session_id: str
    message: str
    doc_type: Optional[str] = None
    regulation_type: Optional[str] = None

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

# ─────────────────────────────────────────────────────────────────────────────
# Routes

@app.get("/health")
async def health():
    if manager.init_error:
        return JSONResponse(status_code=500, content={"status": "error", "detail": manager.init_error})
    if not manager.shared_rag:
        return {"status": "loading", "detail": "RAG is still initializing..."}
    return {"status": "ok", "vectorstore": "ready"}

@app.post("/sessions", response_model=SessionResponse)
async def create_session():
    try:
        session_id = manager.create_session()
        data = manager.get_session(session_id)
        return SessionResponse(
            session_id=session_id, 
            title=data["title"], 
            created_at=data["created_at"]
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

@app.get("/sessions", response_model=List[SessionSummary])
async def list_sessions():
    result = []
    for sid, data in manager.sessions.items():
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
    data = manager.get_session(session_id)
    return SessionDetail(
        session_id=session_id,
        title=data["title"],
        created_at=data["created_at"],
        messages=[MessageItem(**m) for m in data["messages"]],
    )

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    manager.delete_session(session_id)
    return {"status": "deleted"}

@app.delete("/sessions/{session_id}/history")
async def clear_history(session_id: str):
    data = manager.get_session(session_id)
    data["messages"] = []
    data["title"] = "Cuộc trò chuyện mới"
    data["rag"].memory.clear()
    return {"status": "cleared"}

@app.patch("/sessions/{session_id}/title")
async def rename_session(session_id: str, request: Request):
    body = await request.json()
    data = manager.get_session(session_id)
    data["title"] = body.get("title", "Untitled")
    return {"status": "ok"}

@app.post("/chat")
async def chat(req: ChatRequest):
    if not manager.shared_rag:
        raise HTTPException(status_code=503, detail="RAG system is still initializing.")
    
    data = manager.get_session(req.session_id)
    rag: UniversityRAG = data["rag"]
    
    # Track user message
    data["messages"].append({
        "role": "user", 
        "content": req.message, 
        "timestamp": datetime.utcnow().isoformat()
    })

    # Auto-generate title after first message
    if len(data["messages"]) == 1:
        data["title"] = req.message[:50] + ("..." if len(req.message) > 50 else "")

    async def _streamer():
        full_answer = ""
        try:
            async for chunk_dict in rag.astream_query(
                question=req.message,
                doc_type=req.doc_type,
                regulation_type=req.regulation_type,
            ):
                chunk_type = chunk_dict.get("type")
                if chunk_type == "content":
                    full_answer += chunk_dict.get("content", "")
                
                yield json.dumps(chunk_dict, ensure_ascii=False) + "\n"
            
            # Record assistant message
            data["messages"].append({
                "role": "assistant",
                "content": full_answer,
                "timestamp": datetime.utcnow().isoformat(),
            })
        except Exception as e:
            yield f"\n[Lỗi hệ thống: {str(e)}]"

    return StreamingResponse(_streamer(), media_type="text/plain")

# ─────────────────────────────────────────────────────────────────────────────
# Static Assets

FRONTEND = Path(__file__).parent / "index.html"

@app.get("/")
async def serve_frontend():
    if FRONTEND.exists():
        return FileResponse(FRONTEND)
    return {"message": "Place index.html in the same folder as server.py"}

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)