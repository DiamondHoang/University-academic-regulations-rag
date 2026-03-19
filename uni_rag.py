import os
import logging
import re
import time
import asyncio
from typing import List, Optional, Dict, Any, AsyncGenerator

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

try:
    import fcntl
except ImportError:
    # fcntl is not available on Windows, provide a dummy for local dev
    fcntl = None

from config import Config
from memory.conversation_memory import ConversationMemory
from retrieval.vector_retriever import VectorRetriever
from retrieval.response_generator import ResponseGenerator

# Standardized logging configuration
logging.basicConfig(level=logging.ERROR, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class UniversityRAG:
    """Core RAG system for university regulations with modular retrieval and generation components."""
    
    def __init__(
        self, 
        config: Optional[Dict] = None, 
        session_id: Optional[str] = None,
        embeddings: Optional[HuggingFaceEmbeddings] = None,
        response_generator: Optional[ResponseGenerator] = None
    ):
        """Initialize the RAG system.
        
        Args:
            config: Optional configuration overrides.
            session_id: Optional session ID for memory isolation and persistence.
            embeddings: Optional shared embeddings instance to save memory.
            response_generator: Optional shared response generator instance.
        """
        self.config = {**Config.as_dict(), **(config or {})}
        
        # 1. Initialize Embeddings
        self.embeddings = embeddings or HuggingFaceEmbeddings(
            model_name=self.config["embedding_model"]
        )
        
        # 2. Setup Vector Store and Components
        self.vectorstore: Optional[Chroma] = None
        self.retriever: Optional[VectorRetriever] = None
        
        # 3. Initialize Response Generation and Memory
        self.response_generator = response_generator or ResponseGenerator(self.config)
        self.memory = ConversationMemory(
            max_history=self.config["max_history"],
            session_id=session_id
        )
        self.all_chunks: List[Document] = []
    
    def build_vectorstore(self, documents: List[Document], force_rebuild: bool = False) -> None:
        """Initialize or load the vector store with optional concurrency protection.
        
        Args:
            documents: List of pre-loaded Documents.
            force_rebuild: If True, delete and recreate the vector store.
        """
        db_path = self.config["db_path"]
        lock_path = f"{db_path}.lock"
        
        # Ensure directory exists for locking and storage
        os.makedirs(os.path.dirname(lock_path) if os.path.dirname(lock_path) else ".", exist_ok=True)
        
        lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT)
        try:
            # Concurrency Protection: Acquire an exclusive lock (blocks others)
            if fcntl:
                logger.info("Acquiring lock for vectorstore initialization...")
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
                logger.info("Lock acquired.")
            else:
                logger.warning("fcntl not available. Skipping cross-process file lock (not safe for single-worker).")

            # 1. Split documents into manageable chunks
            chunks = self._split_documents(documents)
            self.all_chunks = chunks
            
            # 2. Build or Load ChromaDB
            if force_rebuild or not os.path.exists(db_path):
                logger.info("Building new vectorstore...")
                self.vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=self.embeddings,
                    persist_directory=db_path,
                    collection_metadata=Config.EMBEDDING_KWARGS
                )
            else:
                logger.info("Loading existing vectorstore...")
                self.vectorstore = self._load_chroma_with_retry(db_path)
            
            # 3. Initialize the Hybrid Retriever
            self.retriever = VectorRetriever(self.vectorstore)
            
        finally:
            if fcntl:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                logger.info("Lock released.")
            os.close(lock_fd)
    
    def _load_chroma_with_retry(self, path: str, retries: int = 1) -> Chroma:
        """Helper to load ChromaDB with a simple retry on collision errors."""
        for attempt in range(retries + 1):
            try:
                return Chroma(
                    persist_directory=path,
                    embedding_function=self.embeddings
                )
            except Exception as e:
                if attempt < retries:
                    logger.warning(f"ChromaDB load attempt {attempt+1} failed ({e}), retrying...")
                    time.sleep(2)
                else:
                    raise e

    def _split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks using configuration rules."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config["chunk_size"],
            chunk_overlap=self.config["chunk_overlap"],
            separators=Config.SEPARATORS,
            length_function=len,
        )
        all_chunks = text_splitter.split_documents(documents)
        
        # Assign unique chunk IDs for tracking
        for i, chunk in enumerate(all_chunks):
            chunk.metadata["chunk_id"] = i
        
        return all_chunks
    
    async def aquery(
        self,
        question: str,
        doc_type: Optional[str] = None,
        regulation_type: Optional[str] = None,
        k: Optional[int] = None
    ) -> str:
        """Process a query asynchronously and return the generated answer."""
        try:
            if not self.vectorstore or not self.retriever:
                return "Hệ thống RAG chưa được khởi tạo hoàn toàn (thiếu vectorstore)."
            
            # 1. Query Expansion & Rewriting
            # Use conversation history to make the query standalone
            conv_history = self.memory.get_context_string(include_last_n=3)
            
            if self._should_rewrite_query(question, conv_history):
                search_query = await self.response_generator.rewrite_query(question, conv_history)
            else:
                search_query = question
                
            search_query = self._preprocess_query(search_query)
            
            # 2. Audience & Metadata Detection
            if not doc_type:
                audience = self._detect_audience_heuristics(search_query)
                doc_type = self._auto_detect_doc_type(audience)
            
            # 3. Document Retrieval
            retrieved_docs = await self.retriever.aretrieve(
                search_query,
                k=k or self.config["max_retrieved_docs"],
                doc_type=doc_type,
                regulation_type=regulation_type,
            )
            
            if not retrieved_docs:
                answer = "Tôi không tìm thấy thông tin liên quan trong các quy định hiện có."
                self.memory.add_turn(question, answer)
                return answer
            
            # 4. Filter and Rank
            max_docs = self.config.get("max_response_docs", Config.MAX_RESPONSE_DOCS)
            ranked_docs = retrieved_docs[:max_docs]
            
            # 5. Answer Generation
            try:
                gen_result = await self.response_generator.agenerate(
                    query=question,
                    documents=ranked_docs,
                    conversation_history=self.memory.get_context_string(include_last_n=2),
                )
                answer = gen_result.get("answer", "")
            except Exception as e:
                logger.error(f"Generation failed: {e}")
                answer = "Hệ thống gặp sự cố khi tạo câu trả lời. Vui lòng thử lại sau."
            
            # 6. Update Memory
            self.memory.add_turn_with_data({
                "question": question,
                "answer": answer,
                "documents": ranked_docs,
            })
            
            return answer
            
        except Exception as e:
            logger.error(f"Aquery failed: {e}", exc_info=True)
            return f"Xin lỗi, có lỗi hệ thống xảy ra: {str(e)}"

    async def astream_query(
        self,
        question: str,
        doc_type: Optional[str] = None,
        regulation_type: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Process a query and stream the response tokens."""
        try:
            if not self.vectorstore or not self.retriever:
                yield "Vector store chưa được khởi tạo."
                return
            
            # 1. Prepare Query
            conv_history = self.memory.get_context_string(include_last_n=3)
            
            if self._should_rewrite_query(question, conv_history):
                search_query = await self.response_generator.rewrite_query(question, conv_history)
            else:
                search_query = question
                
            search_query = self._preprocess_query(search_query)
            
            if not doc_type:
                audience = self._detect_audience_heuristics(search_query)
                doc_type = self._auto_detect_doc_type(audience)
            
            # 2. Retrieve
            retrieved_docs = await self.retriever.aretrieve(
                search_query,
                k=self.config["max_retrieved_docs"],
                doc_type=doc_type,
                regulation_type=regulation_type,
            )
            
            if not retrieved_docs:
                yield "Tôi không tìm thấy thông tin liên quan trong các quy định hiện có."
                return
            
            # 3. Stream Generation
            max_docs = self.config.get("max_response_docs", Config.MAX_RESPONSE_DOCS)
            ranked_docs = retrieved_docs[:max_docs]
            
            async for chunk in self.response_generator.astream_generate(
                query=question,
                documents=ranked_docs,
                conversation_history=self.memory.get_context_string(include_last_n=2)
            ):
                yield chunk
                
        except Exception as e:
            logger.error(f"Streaming Query failed: {e}", exc_info=True)
            yield f"\n[Lỗi hệ thống trong khi tạo phản hồi: {str(e)}]"
    
    def _preprocess_query(self, query: str) -> str:
        """Expand common abbreviations to improve search recall."""
        processed = query
        for abbr, expansion in Config.ABBREVIATIONS.items():
            processed = re.sub(rf"\b{abbr}\b", expansion, processed, flags=re.IGNORECASE)
        return processed

    def _should_rewrite_query(self, query: str, history: str) -> bool:
        """Heuristic to decide if a query needs LLM rewriting."""
        if not history:
            return False
            
        # If it's short, it likely needs context (e.g. "Cái đó là gì?", "Điều kiện thế nào?")
        word_count = len(query.split())
        if word_count < 5:
            return True
            
        # If it contains context-dependent pronouns (Vietnamese)
        context_words = ["đó", "này", "kia", "ấy", "họ", "nó", "thế nào", "bao nhiêu", "ai", "đâu"]
        query_lower = query.lower()
        if any(rf"\b{word}\b" in query_lower for word in context_words):
            return True
            
        # If it's long and has specific keywords, it's likely standalone
        standalone_keywords = ["quy định", "đăng ký", "học phần", "tín chỉ", "hồ sơ", "thời hạn"]
        if word_count > 10 and any(kw in query_lower for kw in standalone_keywords):
            return False
            
        # Default: rewrite if in doubt but we have history
        return True

    def _detect_audience_heuristics(self, text: str) -> str:
        """Basic heuristic analysis to detect target audience from query text."""
        text = text.lower()
        if any(w in text for w in ["sinh viên", "sv", "chính quy", "đại học"]):
            return "sinh_vien_chinh_quy"
        if any(w in text for w in ["cao học", "thạc sĩ", "học viên"]):
            return "hoc_vien_cao_hoc"
        if any(w in text for w in ["nghiên cứu sinh", "ncs", "tiến sĩ"]):
            return "nghien_cuu_sinh"
        return "sinh_vien_chinh_quy" # Default fallback
    
    def _auto_detect_doc_type(self, target_audience: Any) -> Optional[str]:
        """Map target audience keywords to internal document category IDs (e.g., DTDH, DTSDH)."""
        # Handle cases where LLM or caller might pass a list
        if isinstance(target_audience, list):
            target_audience = target_audience[0] if target_audience else ""
        
        target_audience = str(target_audience).strip().lower()

        # Normalization map for various ways audience might be identified
        _MAP = {
            "sinh viên": "sinh_vien_chinh_quy",
            "sinh_vien": "sinh_vien_chinh_quy",
            "undergraduate": "sinh_vien_chinh_quy",
            "học viên cao học": "hoc_vien_cao_hoc",
            "cao học": "hoc_vien_cao_hoc",
            "master": "hoc_vien_cao_hoc",
            "nghiên cứu sinh": "nghien_cuu_sinh",
            "tiến sĩ": "nghien_cuu_sinh",
            "phd": "nghien_cuu_sinh",
        }
        
        canonical_key = _MAP.get(target_audience, target_audience)
        return Config.AUDIENCE_MAPPING.get(canonical_key)