import os
from typing import List, Optional, Dict, Any, AsyncGenerator, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from config import Config
from memory.conversation_memory import ConversationMemory
from retrieval.vector_retriever import VectorRetriever
from retrieval.response_generator import ResponseGenerator

# Standard User-Facing Error Messages (Vietnamese)
ERROR_MISSING_DB = "Hệ thống chưa sẵn sàng (thiếu cơ sở dữ liệu vector)."
ERROR_NO_RELEVANT = "Tôi không tìm thấy thông tin liên quan trong các quy định hiện hành."
ERROR_GENERIC = "Hệ thống gặp sự cố khi xử lý câu hỏi. Vui lòng thử lại sau."

class UniversityRAG:
    """Main RAG orchestrator for University Academic Regulations."""
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        embeddings: Optional[HuggingFaceEmbeddings] = None,
        response_generator: Optional[ResponseGenerator] = None
    ):
        self.config = {**Config.as_dict(), **(config or {})}
        
        # 1. Embeddings
        self.embeddings = embeddings or HuggingFaceEmbeddings(
            model_name=self.config["embedding_model"]
        )
        
        # 2. Retrieval & Generation
        self.vectorstore: Optional[Chroma] = None
        self.retriever: Optional[VectorRetriever] = None
        
        self.response_generator = response_generator or ResponseGenerator(self.config)
        self.memory = ConversationMemory(
            max_history=self.config["max_history"],
            session_id=session_id
        )
    
    def build_vectorstore(self, documents: List[Document], force_rebuild: bool = False) -> None:
        """Initialize or rebuild the vector database."""
        db_path = self.config["db_path"]
        chunks = self._split_documents(documents)
        
        if force_rebuild or not os.path.exists(db_path):
            print(f"[RAG] Building vectorstore at {db_path}")
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=db_path,
                collection_metadata=Config.EMBEDDING_KWARGS
            )
        else:
            print(f"[RAG] Loading existing vectorstore from {db_path}")
            self.vectorstore = Chroma(
                persist_directory=db_path,
                embedding_function=self.embeddings
            )
        
        self.retriever = VectorRetriever(self.vectorstore)
    
    def _split_documents(self, documents: List[Document]) -> List[Document]:
        """Chunk documents for ingestion."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config["chunk_size"],
            chunk_overlap=self.config["chunk_overlap"],
            separators=Config.SEPARATORS,
            length_function=len,
        )
        all_chunks = text_splitter.split_documents(documents)
        for i, chunk in enumerate(all_chunks):
            chunk.metadata["chunk_id"] = i
        return all_chunks
    
    async def _prepare_search_query(self, question: str) -> str:
        """Rewrite query based on history if available."""
        conv_history = self.memory.get_context_string(include_last_n=1)
        history_msgs = self.memory.get_history_messages(include_last_n=1)
        
        if conv_history or history_msgs:
            return await self.response_generator.rewrite_query(
                question, 
                conv_history, 
                history_messages=history_msgs
            )
        return question

    async def aquery(
        self,
        question: str,
        k: Optional[int] = None
    ) -> str:
        """Process a question and return a complete answer string."""
        try:
            if not self.vectorstore or not self.retriever:
                return ERROR_MISSING_DB
            
            # 1. Query Preparation
            search_query = await self._prepare_search_query(question)
            
            # 2. Retrieval
            retrieved_docs = await self.retriever.retrieve(
                search_query,
                k=k or self.config["max_retrieved_docs"]
            )
            
            if not retrieved_docs:
                self.memory.add_turn(question, ERROR_NO_RELEVANT)
                return ERROR_NO_RELEVANT
            
            # 3. Generation
            try:
                gen_result = await self.response_generator.agenerate(
                    query=question,
                    documents=retrieved_docs,
                    conversation_history=self.memory.get_context_string(include_last_n=2),
                    history_messages=self.memory.get_history_messages(include_last_n=2),
                )
                answer = gen_result.get("answer", ERROR_GENERIC)
            except Exception as e:
                print(f"[ERROR] [RAG] Generation failed: {e}")
                answer = ERROR_GENERIC
            
            # 4. Memory Update
            self.memory.add_turn_with_data({
                "question": question,
                "answer": answer,
                "documents": retrieved_docs[:Config.MAX_RESPONSE_DOCS],
            })
            
            return answer
            
        except Exception as e:
            print(f"[RAG] Unexpected error in aquery: {e}")
            return ERROR_GENERIC

    async def astream_query(
        self,
        question: str,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Process a question and stream the response."""
        try:
            if not self.vectorstore or not self.retriever:
                yield {"type": "content", "content": ERROR_MISSING_DB}
                return
            
            # 1. Prepare Query
            search_query = await self._prepare_search_query(question)
            
            # 2. Retrieval
            retrieved_docs = await self.retriever.retrieve(
                search_query,
                k=self.config["max_retrieved_docs"]
            )
            
            if not retrieved_docs:
                yield {"type": "content", "content": ERROR_NO_RELEVANT}
                return
            
            # 3. Metadata (Sources)
            ranked_docs = retrieved_docs[:Config.MAX_RESPONSE_DOCS]
            yield {
                "type": "metadata",
                "sources": [{"content": dc.page_content, "metadata": dc.metadata} for dc in ranked_docs]
            }

            # 4. Stream Generation
            full_answer = ""
            async for chunk in self.response_generator.astream_generate(
                query=question,
                documents=ranked_docs,
                conversation_history=self.memory.get_context_string(include_last_n=2),
                history_messages=self.memory.get_history_messages(include_last_n=2)
            ):
                full_answer += chunk
                yield {"type": "content", "content": chunk}
                
            self.memory.add_turn(question, full_answer)
                
        except Exception as e:
            print(f"[RAG] Unexpected error in astream_query: {e}")
            yield {"type": "content", "content": f"\n[{ERROR_GENERIC}]\n"}
