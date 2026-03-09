"""Centralized configuration module"""
import os
from typing import Dict

class Config:
    """Application configuration"""

    # Vector Store Settings
    DB_PATH = os.environ.get("DB_PATH", "vector_db")
    EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")
    EMBEDDING_KWARGS = {"hnsw:space": "cosine"}

    # LLM Settings — cloud models require: docker exec -it ollama ollama signin
    LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-v3.1:671b-cloud")
    LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.1"))

    # Retrieval Settings
    USE_HYBRID_SEARCH = False
    USE_RERANKER = False
    USE_PARENT_CHILD = False
    
    MAX_RETRIEVED_DOCS = 10
    BM25_K = 20
    RRF_K = 60
    MAX_CONTEXT_LENGTH = 8000
    SIMILARITY_THRESHOLD = 0.2
    CROSS_ENCODER_MODEL = os.environ.get("CROSS_ENCODER_MODEL", "BAAI/bge-reranker-v2-m3")
    MAX_CHUNKS_PER_FILE = 1  # Strictly 1 chunk per file to avoid duplicates
    MAX_RESPONSE_DOCS = 1    # Only pass 1 doc to LLM to prevent source mixing
    
    # Text Splitting Settings (Parent-Child Strategy)
    # Text Splitting Settings
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 100
    
    # Keeping these for backward compatibility if needed by other components
    PARENT_CHUNK_SIZE = 2000
    PARENT_CHUNK_OVERLAP = 200
    CHILD_CHUNK_SIZE = 400
    CHILD_CHUNK_OVERLAP = 50
    
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
    MARKDOWN_HEADER_PATTERN = r"^#+\s.*(?:page|Page|PAGE).*$\n?"
    DATE_PATTERNS = [
        r"(?:ngày\s*)?(\d{1,2})\s+(?:tháng|thang)\s*(\d{1,2}(?:\s*\d)?)\s+(?:năm|nam)\s+(\d{2,4})",
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
            "parent_chunk_size": cls.PARENT_CHUNK_SIZE,
            "parent_chunk_overlap": cls.PARENT_CHUNK_OVERLAP,
            "child_chunk_size": cls.CHILD_CHUNK_SIZE,
            "child_chunk_overlap": cls.CHILD_CHUNK_OVERLAP,
            "max_history": cls.MAX_HISTORY,
            "confidence_threshold": cls.CONFIDENCE_THRESHOLD,
            "max_retrieved_docs": cls.MAX_RETRIEVED_DOCS,
            "max_context_length": cls.MAX_CONTEXT_LENGTH,
        }
