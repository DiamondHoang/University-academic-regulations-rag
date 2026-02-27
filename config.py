"""Centralized configuration module"""
from typing import Dict

class Config:
    """Application configuration"""
    
    # Vector Store Settings
    DB_PATH = "vector_db"
    EMBEDDING_MODEL = "BAAI/bge-m3"
    EMBEDDING_KWARGS = {"hnsw:space": "cosine"}
    
    # LLM Settings
    LLM_MODEL = "deepseek-v3.1:671b-cloud" # qwen3-coder:480b-cloud, gpt-oss:120b-cloud, deepseek-v3.1:671b-cloud, qwen2.5:3b
    LLM_TEMPERATURE = 0.1
    
    # Retrieval Settings
    USE_HYBRID = True
    MAX_RETRIEVED_DOCS = 7
    BM25_K = 5
    RRF_K = 60
    MAX_CONTEXT_LENGTH = 8000
    SIMILARITY_THRESHOLD = 0.5
    USE_CROSS_ENCODER_RERANK = True
    CROSS_ENCODER_MODEL = "BAAI/bge-reranker-v2-m3"  # Vietnamese-optimized reranker
    RECENCY_WEIGHT = 0.2
    
    # Text Splitting Settings
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Memory Settings
    MAX_HISTORY = 5
    CONFIDENCE_THRESHOLD = 0.65
    
    # Document Loading Settings
    BASE_PATH = "md"
    
    # Text Splitting Separators
    SEPARATORS = [
        "\n</table>",  # End of HTML table - split after table
        "\n<table",    # Start of HTML table - split before table
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
            "use_hybrid": cls.USE_HYBRID,
            "chunk_size": cls.CHUNK_SIZE,
            "chunk_overlap": cls.CHUNK_OVERLAP,
            "max_history": cls.MAX_HISTORY,
            "confidence_threshold": cls.CONFIDENCE_THRESHOLD,
            "max_retrieved_docs": cls.MAX_RETRIEVED_DOCS,
            "max_context_length": cls.MAX_CONTEXT_LENGTH,
        }
