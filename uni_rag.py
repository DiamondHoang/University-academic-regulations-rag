import os
import re
import time
import asyncio
from typing import List, Optional, Dict, Any, AsyncGenerator, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

try:
    import fcntl
except ImportError:
    pass

from config import Config
from memory.conversation_memory import ConversationMemory
from retrieval.vector_retriever import VectorRetriever
from retrieval.response_generator import ResponseGenerator



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
            # 1. Split documents into manageable chunks
            chunks = self._split_documents(documents)
            
            # 2. Build or Load ChromaDB
            if force_rebuild or not os.path.exists(db_path):
                self.vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=self.embeddings,
                    persist_directory=db_path,
                    collection_metadata=Config.EMBEDDING_KWARGS
                )
            else:
                self.vectorstore = self._load_chroma_with_retry(db_path)
            
            # 3. Initialize the Hybrid Retriever
            self.retriever = VectorRetriever(self.vectorstore)
        
        finally:
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
            retrieved_docs = await self.retriever.retrieve(
                search_query,
                k=k or self.config["max_retrieved_docs"],
                doc_type=doc_type,
                regulation_type=regulation_type
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
                answer = "Hệ thống gặp sự cố khi tạo câu trả lời. Vui lòng thử lại sau."
            
            # 6. Update Memory
            self.memory.add_turn_with_data({
                "question": question,
                "answer": answer,
                "documents": ranked_docs,
            })
            
            return answer
            
        except Exception as e:
            return f"Xin lỗi, có lỗi hệ thống xảy ra: {str(e)}"

    async def astream_query(
        self,
        question: str,
        doc_type: Optional[str] = None,
        regulation_type: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
    # astream_query modified for better user experience
        """Process a query and stream identifying metadata followed by tokens.
        Yields Dicts with either 'type': 'metadata' or 'type': 'content' or 'type': 'log'.
        """
        try:
            if not self.vectorstore or not self.retriever:
                yield {"type": "content", "content": "Vector store chưa được khởi tạo."}
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
            retrieved_docs = await self.retriever.retrieve(
                search_query,
                k=self.config["max_retrieved_docs"],
                doc_type=doc_type,
                regulation_type=regulation_type
            )
            
            if not retrieved_docs:
                yield {"type": "content", "content": "Tôi không tìm thấy thông tin liên quan trong các quy định hiện có."}
                return
            
            # 3. Yield Metadata (Sources)
            max_docs = self.config.get("max_response_docs", Config.MAX_RESPONSE_DOCS)
            ranked_docs = retrieved_docs[:max_docs]
            
            yield {
                "type": "metadata",
                "sources": [
                    {
                        "content": dc.page_content,
                        "metadata": dc.metadata
                    } for dc in ranked_docs
                ]
            }

            # 4. Stream Generation
            async for chunk in self.response_generator.astream_generate(
                query=question,
                documents=ranked_docs,
                conversation_history=self.memory.get_context_string(include_last_n=2)
            ):
                yield {"type": "content", "content": chunk}
                
        except Exception as e:
            yield {"type": "content", "content": f"\n[Lỗi hệ thống trong khi tạo phản hồi: {str(e)}]\n"}

    
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
            
        query_lower = query.lower().strip()
        
        # If it's short, it likely needs context (e.g. "Cái đó là gì?", "Điều kiện thế nào?")
        word_count = len(query_lower.split())
        if word_count < 4:
            return True
            
        # If it's a "yes/no" or very simple continuation
        if query_lower in ["vâng", "đúng", "không", "thế à", "ok", "được"]:
            return True

        # If it contains context-dependent pronouns (Vietnamese)
        context_words = ["đó", "này", "kia", "ấy", "họ", "nó", "thế nào", "bao nhiêu", "ai", "đâu", "vậy", "trên"]
        if any(rf"\b{word}\b" in query_lower for word in context_words):
            return True
            
        # If it's long and has specific keywords, it's definitely standalone - trust the user's input
        standalone_keywords = ["quy định", "đăng ký", "học phần", "tín chỉ", "hồ sơ", "thời hạn", "điểm", "tốt nghiệp", "xét", "hủy"]
        if word_count >= 5 and any(kw in query_lower for kw in standalone_keywords):
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