"""Centralized configuration module"""
import os
from typing import Dict

class Config:
    """Application configuration"""

    # Vector Store Settings
    DB_PATH = os.environ.get("DB_PATH", "vector_db")
    EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")
    EMBEDDING_KWARGS = {"hnsw:space": "cosine"}
    
    # Re-ranker Settings
    RERANKER_MODEL = os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    USE_RERANKER = True
    TOP_K_RERANK = 12  # Reduced from 20 to speed up re-ranking

    # LLM Settings — cloud models require: docker exec -it ollama ollama signin
    LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-v3.1:671b-cloud")
    LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.1"))
    OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

    MAX_RETRIEVED_DOCS = 10
    MAX_CONTEXT_LENGTH = 8000
    SIMILARITY_THRESHOLD = 0.2
    MAX_CHUNKS_PER_FILE = 1  # Strictly 1 chunk per file to avoid duplicates
    MAX_RESPONSE_DOCS = 4    # Reduced from 7 to improve LLM processing speed
    
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    
    # Memory Settings
    MAX_HISTORY = 5
    CONFIDENCE_THRESHOLD = 0.35
    
    # Document Loading Settings
    BASE_PATH = "md"
    
    # Text Splitting Separators
    SEPARATORS = [
        "\n\n",        # Paragraph breaks
        "\n#",         # Markdown headers
        "\n##",        # Sub headers
        "\n###",       # Sub-sub headers
        "\n",          # Line breaks
        ". ",          # Sentences ending with period
        "; ",          # Sentences ending with semicolon
        ": ",          # Sentences ending with colon
        " ",           # Words
    ]
    
    # Regex Patterns
    PAGE_HEADER_PATTERN = r'^## Page \d+.*$\n?'
    PAGE_INFO_PATTERN = r'^#+\s.*(?:page|Page|PAGE).*$\n?'
    DATE_PATTERNS = [
        r"(?:ngày\s*)?(\d{1,2})\s*(?:tháng|thang)\s*(\d{1,2}(?:\s*\d)?)\s*(?:năm|nam)\s*(\d{2,4})",
        r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})",
    ]

    
    # Audience Mapping
    AUDIENCE_MAPPING = {
        "sinh_vien_chinh_quy": "DTDH",
        "hoc_vien_cao_hoc": "DTSDH",
        "nghien_cuu_sinh": "DTSDH",
    }

    
    # Abbreviation Mapping
    ABBREVIATIONS = {
        "sv": "sinh viên",
        "tc": "tín chỉ",
        "hk": "học kỳ",
        "ctdt": "chương trình đào tạo",
        "ttnt": "thực tập ngoài trường",
        "đacn": "đồ án chuyên ngành",
        "lvtn": "luận văn tốt nghiệp",
        "kltn": "khóa luận tốt nghiệp",
        "đatn": "đồ án tốt nghiệp",
    }
    
    @classmethod
    def get_markdown_headers(cls) -> list:
        """Get markdown headers for splitting"""
        return [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
        ]
    
    @classmethod
    def as_dict(cls) -> Dict:
        """Convert config to dictionary"""
        return {
            "db_path": cls.DB_PATH,
            "embedding_model": cls.EMBEDDING_MODEL,
            "llm_model": cls.LLM_MODEL,
            "chunk_size": cls.CHUNK_SIZE,
            "chunk_overlap": cls.CHUNK_OVERLAP,
            "max_history": cls.MAX_HISTORY,
            "confidence_threshold": cls.CONFIDENCE_THRESHOLD,
            "max_retrieved_docs": cls.MAX_RETRIEVED_DOCS,
            "max_context_length": cls.MAX_CONTEXT_LENGTH,
            "ollama_base_url": cls.OLLAMA_BASE_URL,
        }
