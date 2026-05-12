"""Centralized configuration module for University Academic Regulations RAG"""
import os
from typing import Dict, List, Tuple

class Config:
    """Application configuration organized by component"""

    # --- Vector Store & Embeddings ---
    DB_PATH = "vector_db"
    EMBEDDING_MODEL = "BAAI/bge-m3"
    EMBEDDING_KWARGS = {"hnsw:space": "cosine"}
    
    # --- Retrieval & Re-ranking ---
    USE_RERANKER = False
    RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
    MAX_RETRIEVED_DOCS = 15
    MAX_RESPONSE_DOCS = 4
    
    # --- Contextual Retrieval (Improved Accuracy) ---
    USE_CONTEXTUAL_RETRIEVAL = True
    CONTEXTUAL_PROMPT = """
    Bạn là chuyên gia tư vấn quy chế đại học. 
    Dưới đây là một đoạn trích từ tài liệu: "{title}".
    Hãy tóm tắt bối cảnh của tài liệu này trong tối đa 20 từ để giúp đoạn trích dưới đây có thể hiểu độc lập.
    
    Đoạn trích: {chunk}
    
    Trả về CHỈ câu tóm tắt bối cảnh, không giải thích gì thêm.
    Kết quả mong muốn (Ví dụ): "Trong quy định về học bổng khuyến khích HK241, quy trình xét duyệt là..."
    """
    
    # --- LLM Providers (Groq / Ollama) ---
    LLM_PROVIDER = "groq" 
    LLM_TEMPERATURE = 0.1
    
    # Groq Settings
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
    GROQ_MODEL = "llama-3.3-70b-versatile"
    
    # Ollama Settings
    OLLAMA_BASE_URL = "http://localhost:11434"
    OLLAMA_MODEL = "deepseek-v3.1:671b-cloud"
    
    # --- Memory & Stability ---
    MAX_HISTORY = 5
    CONFIDENCE_THRESHOLD = 0.2
    
    # --- Document Processing (CRITICAL: Do not remove) ---
    BASE_PATH = "md"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Text Splitting Separators
    SEPARATORS = [
        "\n\n", "\n#", "\n##", "\n###", "\n", ". ", "; ", ": ", " "
    ]
    
    # Regex Patterns for Cleaning (Preserved for doc_loader.py)
    PAGE_HEADER_PATTERN = r'^## Page \d+.*$\n?'
    PAGE_INFO_PATTERN = r'^#+\s.*(?:page|Page|PAGE).*$\n?'
    DATE_PATTERNS = [
        r"(?:ngày\s*)?(\d{1,2})\s*(?:tháng|thang)\s*(\d{1,2}(?:\s*\d)?)\s*(?:năm|nam)\s*(\d{2,4})",
        r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})",
    ]

    @classmethod
    def as_dict(cls) -> Dict:
        """Convert runtime configuration to dictionary for system initialization"""
        return {
            "db_path": cls.DB_PATH,
            "embedding_model": cls.EMBEDDING_MODEL,
            "llm_provider": cls.LLM_PROVIDER,
            "groq_api_key": cls.GROQ_API_KEY,
            "groq_model": cls.GROQ_MODEL,
            "ollama_base_url": cls.OLLAMA_BASE_URL,
            "ollama_model": cls.OLLAMA_MODEL,
            "llm_temperature": cls.LLM_TEMPERATURE,
            "chunk_size": cls.CHUNK_SIZE,
            "chunk_overlap": cls.CHUNK_OVERLAP,
            "max_history": cls.MAX_HISTORY,
            "confidence_threshold": cls.CONFIDENCE_THRESHOLD,
            "max_retrieved_docs": cls.MAX_RETRIEVED_DOCS,
            "max_response_docs": cls.MAX_RESPONSE_DOCS,
        }
