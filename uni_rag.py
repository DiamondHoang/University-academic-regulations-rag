import os
import logging
from typing import List, Optional, Dict

from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.documents import Document

from config import Config
from memory.conversation_memory import ConversationMemory
from retrieval.query_analyzer import QueryAnalyzer
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
        self.query_analyzer = QueryAnalyzer(self.llm)
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
        
        self.retriever = HybridRetriever(self.vectorstore, self.config["use_hybrid"])
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
        all_chunks: List[Document] = []
        
        for doc in cleaned_documents:
            # Split by markdown headers first
            header_chunks = self._split_by_markdown_headers(doc)
            
            # Then apply regular text splitting
            for chunk in header_chunks:
                if len(chunk.page_content) <= self.config["chunk_size"]:
                    all_chunks.append(chunk)
                else:
                    sub_chunks = self._regular_text_split(chunk)
                    all_chunks.extend(sub_chunks)
        
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
    
    
    def _regular_text_split(self, doc: Document) -> List[Document]:
        """Split text using recursive character splitter with table-aware separators
        
        Args:
            doc: Document to split
            
        Returns:
            List of document chunks
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config["chunk_size"],
            chunk_overlap=self.config["chunk_overlap"],
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
        """Process a query and return answer
        
        Args:
            question: User question
            doc_type: Optional document type filter
            regulation_type: Optional regulation type filter
            k: Number of results to retrieve
            
        Returns:
            Answer string with sources
        """
        try:
            if not self.vectorstore:
                return "Vector store chưa được khởi tạo."
            
            conversation_history = self.memory.get_history()
            intent = self.query_analyzer.analyze(question, conversation_history)
            
            confidence = intent.get("confidence", 1.0)
            clarification = ""
            if confidence < self.config["confidence_threshold"]:
                clarification = " (Lưu ý: Phân tích câu hỏi có độ tin cậy thấp)"
            
            search_query = intent["enhanced_query"]
            
            if not doc_type:
                doc_type = self._auto_detect_doc_type(intent["target_audience"])
            
            filters = intent.get("filters", {})
            doc_type = filters.get("doc_type") or doc_type
            regulation_type = filters.get("regulation_type") or regulation_type
            
            retrieved_docs = self.retriever.retrieve(
                search_query,
                k=self.config["max_retrieved_docs"],
                doc_type=doc_type,
                regulation_type=regulation_type,
            )
            
            if not retrieved_docs:
                answer = "Tôi không tìm thấy thông tin liên quan."
                self.memory.add_turn(question, answer)
                return answer
            
            ranked_docs = retrieved_docs[:3]
            conv_history = self.memory.get_context_string(include_last_n=2)
            gen_result = self.response_generator.generate(
                query=question,
                documents=ranked_docs,
                conversation_history=conv_history,
                analysis=intent,
                clean_mode=False,
            )

            # ResponseGenerator returns a dict with 'answer', 'confidence', 'sources'
            answer = gen_result.get("answer", "") + clarification
            confidence = gen_result.get("confidence", 0.0)
            
            turn_data = {
                "question": question,
                "answer": answer,
                "documents": ranked_docs,
                "analysis": intent,
            }
            self.memory.add_turn_with_data(turn_data)
            
            return answer
            
        except Exception as e:
            logger.error(f"Query failed: {e}", exc_info=True)
            error_answer = f"Xin lỗi, có lỗi xảy ra: {str(e)}"
            self.memory.add_turn(question, error_answer)
            return error_answer
    
    def _clean_page_headers(self, content: str) -> str:
        """Remove page headers from markdown content
        
        Args:
            content: Document content
            
        Returns:
            Cleaned content
        """
        import re
        cleaned_content = re.sub(Config.PAGE_HEADER_PATTERN, '', content, flags=re.MULTILINE)
        cleaned_content = re.sub(Config.PAGE_INFO_PATTERN, '', cleaned_content, flags=re.MULTILINE)
        return cleaned_content
    
    def _auto_detect_doc_type(self, target_audience: str) -> Optional[str]:
        """Auto-detect doc_type from target audience
        
        Args:
            target_audience: Target audience identifier
            
        Returns:
            Document type or None
        """
        return Config.AUDIENCE_MAPPING.get(target_audience)
    
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
    rag = UniversityRAG(config={"use_hybrid": True})
    documents = loader.load_documents()
    rag.build_vectorstore(documents, force_rebuild=False)
    rag.chat_loop()


if __name__ == "__main__":
    main()