import os
import logging
import uuid
from typing import List, Optional, Dict
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.documents import Document

from config import Config
from memory.conversation_memory import ConversationMemory
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.response_generator import ResponseGenerator
from loader.doc_loader import RegulationDocumentLoader

logging.basicConfig(level=logging.ERROR, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class UniversityRAG:
    """RAG system for university regulations with modular components"""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize RAG system with configuration
        
        Args:
            config: Optional configuration dictionary to override defaults
        """
        self.config = {**Config.as_dict(), **(config or {})}
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config["embedding_model"]
        )
        self.llm = ChatOllama(
            model=self.config["llm_model"],
            temperature=Config.LLM_TEMPERATURE,
            timeout=120.0  # Tăng timeout cho Ollama
        )
        self.vectorstore: Optional[Chroma] = None
        
        # Initialize modular components
        self.retriever: Optional[HybridRetriever] = None
        self.response_generator = ResponseGenerator(self.llm)
        self.memory = ConversationMemory(
            max_history=self.config["max_history"]
        )
        self.all_chunks: List[Document] = []
    
    def build_vectorstore(self, documents: List[Document], force_rebuild: bool = False) -> None:
        """Build vector store and initialize retriever
        
        Args:
            documents: List of documents to embed
            force_rebuild: Force rebuild even if store exists
        """
        chunks = self._split_documents(documents)
        self.all_chunks = chunks
        
        if force_rebuild or not os.path.exists(self.config["db_path"]):
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.config["db_path"],
                collection_metadata=Config.EMBEDDING_KWARGS
            )
        else:
            self.vectorstore = Chroma(
                persist_directory=self.config["db_path"],
                embedding_function=self.embeddings
            )
        
        self.retriever = HybridRetriever(self.vectorstore)
        self.retriever.build_bm25(chunks)
    
    def _split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks
        
        Args:
            documents: Documents to split
            
        Returns:
            List of document chunks with metadata
        """
        # Clean documents by removing page headers
        cleaned_documents = [
            Document(
                page_content=self._clean_page_headers(doc.page_content),
                metadata=doc.metadata
            )
            for doc in documents
        ]
        
        # Process documents with text splitting
        if not self.config.get("use_parent_child", Config.USE_PARENT_CHILD):
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.get("chunk_size", Config.CHUNK_SIZE),
                chunk_overlap=self.config.get("chunk_overlap", Config.CHUNK_OVERLAP),
                separators=Config.SEPARATORS,
                length_function=len,
            )
            all_chunks = text_splitter.split_documents(cleaned_documents)
        else:
            # Legacy Parent-Child logic
            all_chunks: List[Document] = []
            for doc in cleaned_documents:
                header_chunks = self._split_by_markdown_headers(doc)
                for chunk in header_chunks:
                    parent_chunks = self._parent_text_split(chunk)
                    for p_chunk in parent_chunks:
                        parent_id = str(uuid.uuid4())
                        parent_content = p_chunk.page_content
                        child_chunks = self._child_text_split(p_chunk)
                        for c_chunk in child_chunks:
                            c_chunk.metadata["parent_id"] = parent_id
                            c_chunk.metadata["parent_content"] = parent_content
                            all_chunks.append(c_chunk)
        
        # Add chunk IDs
        for i, chunk in enumerate(all_chunks):
            chunk.metadata["chunk_id"] = i
        
        return all_chunks
    
    def _split_by_markdown_headers(self, doc: Document) -> List[Document]:
        """Split document by markdown headers
        
        Args:
            doc: Document to split
            
        Returns:
            List of document chunks
        """
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=Config.get_markdown_headers(),
            strip_headers=False
        )
        
        try:
            splits = markdown_splitter.split_text(doc.page_content)
            header_chunks = []
            for split in splits:
                new_metadata = doc.metadata.copy()
                new_metadata.update(split.metadata)
                header_chunks.append(Document(
                    page_content=split.page_content,
                    metadata=new_metadata
                ))
            return header_chunks
        except Exception:
            return [doc]
    
    
    def _parent_text_split(self, doc: Document) -> List[Document]:
        """Split text using recursive character splitter to create Parent chunks"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config["parent_chunk_size"],
            chunk_overlap=self.config["parent_chunk_overlap"],
            separators=Config.SEPARATORS,
            length_function=len,
        )
        return text_splitter.split_documents([doc])

    def _child_text_split(self, doc: Document) -> List[Document]:
        """Split parent text using recursive character splitter to create Child chunks"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config["child_chunk_size"],
            chunk_overlap=self.config["child_chunk_overlap"],
            separators=Config.SEPARATORS,
            length_function=len,
        )
        return text_splitter.split_documents([doc])
    
    def query(
        self,
        question: str,
        doc_type: Optional[str] = None,
        regulation_type: Optional[str] = None,
        k: int = 5
    ) -> str:
        """Process a query and return answer with caching and robust fallbacks."""
        try:
            if not self.vectorstore:
                return "Vector store chưa được khởi tạo."
            
            # Preprocess question: expand abbreviations
            processed_question = self._preprocess_query(question)
            
            conversation_history = self.memory.get_history()
            
            # 1. Expand query and detect filters via heuristics
            search_query = self._preprocess_query(question)
            
            # Simple heuristic audience detection if doc_type not provided
            if not doc_type:
                audience = self._detect_audience_heuristics(question)
                doc_type = self._auto_detect_doc_type(audience)
            
            # 2. Hybrid Retrieval
            try:
                retrieved_docs = self.retriever.retrieve(
                    search_query,
                    k=self.config["max_retrieved_docs"],
                    doc_type=doc_type,
                    regulation_type=regulation_type,
                )
            except Exception as e:
                logger.error(f"Retrieval failed: {e}")
                return "Xin lỗi, tôi gặp sự cố khi tìm kiếm thông tin văn bản."
            
            if not retrieved_docs:
                answer = "Tôi không tìm thấy thông tin liên quan trong các quy định hiện có."
                self.memory.add_turn(question, answer)
                return answer
            
            # 4. Response Generation with Fallback
            max_docs = self.config.get("max_response_docs", Config.MAX_RESPONSE_DOCS)
            ranked_docs = retrieved_docs[:max_docs]
            conv_history = self.memory.get_context_string(include_last_n=2)
            
            try:
                gen_result = self.response_generator.generate(
                    query=question,
                    documents=ranked_docs,
                    conversation_history=conv_history,
                    analysis={}, # No longer using query analysis
                    clean_mode=False,
                )
                answer = gen_result.get("answer", "")
            except Exception as e:
                logger.error(f"Generation failed: {e}")
                # Fallback: Just return titles of found documents if generation fails
                found_titles = list(set([d.metadata.get("title", "Không rõ tiêu đề") for d in ranked_docs]))
                answer = f"Tôi đã tìm thấy thông tin trong các văn bản sau nhưng gặp lỗi khi tóm tắt: {', '.join(found_titles)}. Vui lòng thử lại sau."
            
            # Update memory and cache
            turn_data = {
                "question": question,
                "answer": answer,
                "documents": ranked_docs,
            }
            self.memory.add_turn_with_data(turn_data)
            
            return answer
            
        except Exception as e:
            logger.error(f"Global Query failed: {e}", exc_info=True)
            error_answer = f"Xin lỗi, có lỗi hệ thống xảy ra: {str(e)}"
            return error_answer
    
    def _preprocess_query(self, query: str) -> str:
        """Preprocess user query to expand common university abbreviations.
        
        Args:
            query: User's original query
            
        Returns:
            Preprocessed query with expanded terms
        """
        processed = query
        # Use word boundaries to avoid partial matching
        for abbr, expansion in Config.ABBREVIATIONS.items():
            pattern = rf"\b{abbr}\b"
            processed = re.sub(pattern, expansion, processed, flags=re.IGNORECASE)
        
        return processed

    def _clean_page_headers(self, content: str) -> str:
        """Remove page headers from markdown content
        
        Args:
            content: Document content
            
        Returns:
            Cleaned content
        """
        cleaned_content = re.sub(Config.PAGE_HEADER_PATTERN, '', content, flags=re.MULTILINE)
        cleaned_content = re.sub(Config.PAGE_INFO_PATTERN, '', cleaned_content, flags=re.MULTILINE)
        return cleaned_content
    
    def _detect_audience_heuristics(self, text: str) -> str:
        """Heuristic-based audience detection moved from QueryAnalyzer."""
        text = text.lower()
        if any(w in text for w in ["sinh viên", "sv", "chính quy", "đại học"]):
            return "sinh_vien_chinh_quy"
        if any(w in text for w in ["cao học", "thạc sĩ", "học viên"]):
            return "hoc_vien_cao_hoc"
        if any(w in text for w in ["nghiên cứu sinh", "ncs", "tiến sĩ"]):
            return "nghien_cuu_sinh"
        return "sinh_vien_chinh_quy" # Default to undergraduate
    
    def _auto_detect_doc_type(self, target_audience) -> Optional[str]:
        """Auto-detect doc_type from target audience.

        The LLM may return target_audience as a list instead of a plain string,
        so we normalise it first.  We also map common Vietnamese audience phrases
        to the canonical keys used in AUDIENCE_MAPPING.

        Args:
            target_audience: Target audience identifier (str or list).

        Returns:
            Document type or None.
        """
        # Unwrap list → take the first element
        if isinstance(target_audience, list):
            target_audience = target_audience[0] if target_audience else ""

        # Normalise to str just in case
        if not isinstance(target_audience, str):
            target_audience = str(target_audience)

        # Map common plain-language values to AUDIENCE_MAPPING keys
        _NORMALISE = {
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
        normalised = _NORMALISE.get(target_audience.strip().lower(), target_audience)

        return Config.AUDIENCE_MAPPING.get(normalised)
    
    def chat_loop(self) -> None:
        """Interactive chat loop for user interaction"""
        print("=" * 70)
        print("CHATBOT QUY ĐỊNH")
        print("=" * 70)
        print("\nCác lệnh có sẵn:")
        print("  - 'exit'/'quit': Thoát ứng dụng")
        print("  - 'clear': Xóa lịch sử hội thoại")
        print("  - 'history': Xem lịch sử hội thoại")
        print()
        
        current_filter = {"doc_type": None, "regulation_type": None}
        
        while True:
            try:
                user_input = input("\nBạn: ").strip()
                
                if user_input.lower() in ["exit", "quit"]:
                    print("\nCảm ơn bạn đã sử dụng chatbot!")
                    break
                
                if user_input.lower() == "clear":
                    self.memory.clear()
                    print("Đã xóa lịch sử hội thoại")
                    continue
                
                if user_input.lower() == "history":
                    self._show_history()
                    continue
                
                if not user_input:
                    continue
                
                answer = self.query(
                    user_input,
                    doc_type=current_filter["doc_type"],
                    regulation_type=current_filter["regulation_type"]
                )
                print(f"\nBot:\n{answer}")
                
            except KeyboardInterrupt:
                print("\n\nĐã dừng chatbot.")
                break
            except Exception as e:
                logger.error(f"Chat loop error: {e}", exc_info=True)
                print(f"\nLỗi: {e}. Vui lòng thử lại.")
    
    def _show_history(self) -> None:
        """Display conversation history"""
        if not self.memory.history:
            print("Chưa có lịch sử hội thoại")
        else:
            print("\n" + "=" * 60)
            print("LỊCH SỬ HỘI THOẠI")
            print("=" * 60)
            for i, turn in enumerate(self.memory.history, 1):
                print(f"\n{i}. Câu hỏi: {turn['question']}")
                preview = turn['answer'][:200] + "..." if len(turn['answer']) > 200 else turn['answer']
                print(f"   Trả lời: {preview}")
            print("=" * 60)


def main() -> None:
    """Main entry point for the application"""
    loader = RegulationDocumentLoader(base_path=Config.BASE_PATH)
    rag = UniversityRAG()
    documents = loader.load_documents()
    rag.build_vectorstore(documents, force_rebuild=False)
    rag.chat_loop()


if __name__ == "__main__":
    main()