"""Centralized configuration module for University Academic Regulations RAG"""
import os
from typing import Dict, List, Tuple

class Config:
    """Application configuration organized by component"""

    # --- Vector Store Settings ---
    DB_PATH = os.environ.get("DB_PATH", "vector_db")
    EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")
    EMBEDDING_KWARGS = {"hnsw:space": "cosine"}
    
    # --- Retrieval & Re-ranking Settings ---
    USE_RERANKER = os.environ.get("USE_RERANKER", "True").lower() == "true"
    RERANKER_MODEL = os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    MAX_RETRIEVED_DOCS = int(os.environ.get("MAX_RETRIEVED_DOCS", "10"))
    
    # --- LLM Settings (Ollama) ---
    OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "deepseek-v3.1:671b-cloud")
    
    LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.1"))
    
    # --- Context & Response Settings ---
    MAX_RESPONSE_DOCS = int(os.environ.get("MAX_RESPONSE_DOCS", "3"))    # Documents to include in LLM prompt
    
    # --- Memory & Stability ---
    MAX_HISTORY = 5
    CONFIDENCE_THRESHOLD = 0.35
    
    # --- Document Processing ---
    BASE_PATH = "md"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Text Splitting Separators (ordered by priority)
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
    
    # Regex Patterns for Cleaning
    PAGE_HEADER_PATTERN = r'^## Page \d+.*$\n?'
    PAGE_INFO_PATTERN = r'^#+\s.*(?:page|Page|PAGE).*$\n?'
    DATE_PATTERNS = [
        r"(?:ngày\s*)?(\d{1,2})\s*(?:tháng|thang)\s*(\d{1,2}(?:\s*\d)?)\s*(?:năm|nam)\s*(\d{2,4})",
        r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})",
    ]

    # --- Domain-specific Mappings ---
    
    # Maps internal audience IDs to university acronyms
    AUDIENCE_MAPPING = {
        "sinh_vien_chinh_quy": "DTDH", # Đào tạo Đại học
        "hoc_vien_cao_hoc": "DTSDH",  # Đào tạo Sau đại học
        "nghien_cuu_sinh": "DTSDH",
    }

    # Common university abbreviations for query expansion
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
    def get_markdown_headers(cls) -> List[Tuple[str, str]]:
        """Get markdown headers for splitting"""
        return [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
        ]
    
    @classmethod
    def as_dict(cls) -> Dict:
        """Convert runtime configuration to dictionary for system initialization"""
        return {
            "db_path": cls.DB_PATH,
            "embedding_model": cls.EMBEDDING_MODEL,
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
