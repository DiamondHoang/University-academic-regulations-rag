import logging
import asyncio
import re
import unicodedata
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from langchain_core.documents import Document
from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder
from config import Config

logger = logging.getLogger(__name__)

class VectorRetriever:
    """Simplified Vector-based retriever for university regulations."""

    def __init__(self, vectorstore: Chroma):
        """Initialize retriever"""
        self.vectorstore = vectorstore
        self._date_cache: Dict[str, float] = {}  # Cache for parsed timestamps
        
        # Initialize Cross-Encoder for re-ranking
        if Config.USE_RERANKER:
            try:
                logger.info(f"Loading Re-ranker model: {Config.RERANKER_MODEL}")
                self.reranker = CrossEncoder(Config.RERANKER_MODEL)
            except Exception as e:
                logger.error(f"Failed to load Re-ranker: {e}")
                self.reranker = None
        else:
            self.reranker = None

    def build_bm25(self, chunks: List[Document]) -> None:
        """Deprecated: Logic removed."""
        pass

    async def aretrieve(
        self,
        query: str,
        k: int = 5,
        doc_type: Optional[str] = None,
        regulation_type: Optional[str] = None,
    ) -> List[Document]:
        """Retrieve documents using vector search + cross-encoder re-ranking."""
        try:
            # 1. Search Stage (Vector only)
            # Fetch more candidates for re-ranking
            fetch_k = Config.TOP_K_RERANK if self.reranker else k * 2
            retrieved_docs = await self._avector_only_search(query, fetch_k, doc_type, regulation_type)
            
            if not retrieved_docs and (doc_type or regulation_type):
                logger.warning(f"Filtered retrieval returned 0 results. Falling back to unfiltered.")
                retrieved_docs = await self._avector_only_search(query, fetch_k, None, None)

            if not retrieved_docs:
                return []

            # 2. Re-ranking Stage (Cross-Encoder)
            # Skip re-ranking if the top result is already very confident (Fast-path)
            if self.reranker and len(retrieved_docs) > 1:
                top_score = retrieved_docs[0].metadata.get("confidence_score", 0.0)
                if top_score > 0.75:
                    logger.info(f"High confidence ({top_score:.2f}) - Skipping re-ranking fast-path.")
                else:
                    try:
                        # Prepare pairs for cross-encoder - Truncate content slightly to speed up inference
                        pairs = [[query, doc.page_content[:700]] for doc in retrieved_docs]
                        
                        # Run CPU-intensive re-ranking with batching
                        scores = await asyncio.to_thread(
                            self.reranker.predict, 
                            pairs, 
                            batch_size=16
                        )
                        
                        # Update scores in metadata
                        for i, score in enumerate(scores):
                            retrieved_docs[i].metadata["rerank_score"] = float(score)
                            # Blend scores or replace? Let's use re-ranker score as the primary relevance signal
                            retrieved_docs[i].metadata["confidence_score"] = round(float(score), 4)

                        # Sort by re-rank score
                        retrieved_docs.sort(key=lambda x: x.metadata["rerank_score"], reverse=True)
                    except Exception as e:
                        logger.error(f"Re-ranking failed: {e}")

            # 3. Conflict Resolution - keep only the NEWEST doc per regulation topic
            # Note: We do this AFTER re-ranking to ensure we don't discard a relevant newer doc 
            # or keep an irrelevant older one based only on vector score.
            retrieved_docs = self._resolve_conflicts_by_date(retrieved_docs)

            return retrieved_docs[:k]
        except Exception as e:
            logger.error(f"Retrieval pipeline failed: {e}")
            return []

    async def _avector_only_search(self, query: str, k: int, doc_type: Optional[str], regulation_type: Optional[str]) -> List[Document]:
        """Pure vector search with metadata filtering, using real relevance scores."""
        candidate_k = int(k * 1.5)
        
        chroma_filter = {}
        if doc_type:
            chroma_filter["doc_type"] = doc_type
        if regulation_type:
            chroma_filter["regulation_type"] = regulation_type

        vector_kwargs = {"k": candidate_k}
        if chroma_filter:
            conditions = []
            for k_filt, v_filt in chroma_filter.items():
                if isinstance(v_filt, list):
                    conditions.append({k_filt: {"$in": v_filt}})
                else:
                    conditions.append({k_filt: v_filt})
            
            if len(conditions) == 1:
                vector_kwargs["filter"] = conditions[0]
            else:
                vector_kwargs["filter"] = {"$and": conditions}

        try:
            # Vector search with relevance scores - using asyncio.to_thread
            # which automatically handles the event loop and thread pool
            scored = await asyncio.to_thread(
                self.vectorstore.similarity_search_with_relevance_scores,
                query, 
                **vector_kwargs
            )
            docs = []
            for doc, score in scored:
                doc.metadata["confidence_score"] = round(float(score), 4)
                docs.append(doc)
            return docs
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Vector search failed: {error_msg}")
            
            # Specialized handling for ChromaDB "Error finding id" or plan execution errors
            if "Error finding id" in error_msg or "Internal error" in error_msg:
                logger.warning("Detected potential ChromaDB index issue. Attempting basic similarity search fallback.")
                try:
                    return self.vectorstore.similarity_search(query, **vector_kwargs)
                except Exception as inner_e:
                    logger.error(f"Fallback similarity search also failed: {inner_e}")
            
            return []

    def _parse_date(self, date_str: Any) -> float:
        """Parse Vietnamese date strings to timestamp for comparison."""
        if not date_str:
            return 0.0
        
        date_str = str(date_str).strip()
        if date_str in self._date_cache:
            return self._date_cache[date_str]
        
        try:
            # Handle YYYY
            if len(date_str) == 4 and date_str.isdigit():
                t = datetime.strptime(date_str, "%Y").timestamp()
            # Handle DD/MM/YYYY or DD-MM-YYYY
            else:
                clean_date = date_str.replace("/", "-")
                parts = clean_date.split("-")
                if len(parts) == 3:
                    if len(parts[0]) == 4: # YYYY-MM-DD
                        t = datetime.strptime(clean_date, "%Y-%m-%d").timestamp()
                    else: # DD-MM-YYYY
                        t = datetime.strptime(clean_date, "%d-%m-%Y").timestamp()
                else:
                    t = 0.0
            
            self._date_cache[date_str] = t
            return t
        except Exception:
            return 0.0

    def _resolve_conflicts_by_date(self, docs: List[Document]) -> List[Document]:
        """Resolve regulation conflicts by keeping only the NEWEST document per topic group."""

        def _normalize_title(title: str) -> str:
            # 1. Base normalization: lowercase and NFKD to decompose accents
            t = title.lower().strip()
            t = unicodedata.normalize('NFKD', t)
            # Remove all non-spacing marks (accents) to handle different encodings
            t = "".join([c for c in t if not unicodedata.combining(c)])
            
            # 2. Hardcode strong grouping for versioned regulation series 
            # (Ensures all editions fall into the exact same bucket regardless of noisy suffixes)
            t = t.replace('hdhv', 'hoi dong hoc vu')
            if 'hoi dong hoc vu' in t:
                return 'hoi dong hoc vu'
            if 'quy che dao tao' in t:
                return 'quy che dao tao'
                
            # 3. Strip leading numbers/delimiters
            t = re.sub(r'^\s*[\d\.\-\_]+', '', t)
            
            # 4. Standardize common abbreviations
            t = t.replace('dh&sdh', 'dai hoc va sau dai hoc')
            t = t.replace('dh sdh', 'dai hoc va sau dai hoc')
            t = t.replace('tb', 'thong bao')
            t = t.replace('kl', 'ket luan')
            
            # 5. Remove common descriptive suffixes & file extensions
            t = t.replace('.md', '')
            t = t.replace('signed', '')
            t = t.replace('phien ban hop nhat', '')
            
            # Final cleanup
            t = re.sub(r'[^a-z0-9\s]', ' ', t)
            t = re.sub(r'\s+', ' ', t).strip()
            return t

        groups_best_ts: Dict[str, float] = {}

        # Pass 1: Find the maximum (newest) timestamp for each topic key
        for doc in docs:
            title = doc.metadata.get("title", doc.metadata.get("file_path", ""))
            key = _normalize_title(title)
            ts = self._parse_date(doc.metadata.get("issue_date"))
            
            if key not in groups_best_ts:
                groups_best_ts[key] = ts
            else:
                if ts > groups_best_ts[key]:
                    groups_best_ts[key] = ts

        # Pass 2: Keep ALL chunks that belong to the newest edition for their topic key
        resolved: List[Document] = []
        for doc in docs:
            title = doc.metadata.get("title", doc.metadata.get("file_path", ""))
            key = _normalize_title(title)
            ts = self._parse_date(doc.metadata.get("issue_date"))
            
            if ts == groups_best_ts[key]:
                resolved.append(doc)
                
        return resolved
