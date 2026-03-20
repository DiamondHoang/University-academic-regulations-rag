import asyncio
import re
import unicodedata
from typing import List, Optional, Dict, Any
from datetime import datetime
from langchain_core.documents import Document
from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder
from config import Config

class VectorRetriever:
    """Hybrid retriever combining Vector Search and BM25 for university regulations."""

    def __init__(self, vectorstore: Chroma):
        """Initialize retriever"""
        self.vectorstore = vectorstore
        self._date_cache: Dict[str, float] = {}  # Cache for parsed timestamps
        
        # Initialize Cross-Encoder for re-ranking (if enabled)
        if Config.USE_RERANKER:
            try:
                self.reranker = CrossEncoder(Config.RERANKER_MODEL)
            except Exception as e:
                self.reranker = None
        else:
            self.reranker = None


    async def retrieve(
        self,
        query: str,
        k: int = 5,
        doc_type: Optional[str] = None,
        regulation_type: Optional[str] = None
    ) -> List[Document]:
        """Retrieve documents using Vector Search and Re-ranking."""
        try:
            # 1. Retrieval Stage - Vector Search
            vector_docs = await self._vector_search(query, k * 2, doc_type, regulation_type)
            
            # 2. Results Stage
            retrieved_docs = vector_docs
            
            # Limit to a safe number for re-ranking and resolution
            retrieved_docs = retrieved_docs[:k * 2]

            # 3. Re-ranking Stage (Cross-Encoder) - Only if enabled
            if self.reranker and len(retrieved_docs) > 1:
                try:
                    # Prepare pairs for cross-encoder
                    pairs = [[query, doc.page_content[:700]] for doc in retrieved_docs]
                    scores = await asyncio.to_thread(
                        self.reranker.predict, 
                        pairs, 
                        batch_size=16, 
                        show_progress_bar=False
                    )
                    
                    for i, score in enumerate(scores):
                        retrieved_docs[i].metadata["rerank_score"] = float(score)
                        retrieved_docs[i].metadata["confidence_score"] = round(float(score), 4)

                    retrieved_docs.sort(key=lambda x: x.metadata["rerank_score"], reverse=True)
                except Exception:
                    pass

            # 4. Conflict Resolution - keep only the NEWEST doc per regulation topic
            retrieved_docs = self._resolve_conflicts_by_date(retrieved_docs)

            return retrieved_docs[:k]
        except Exception:
            return []

    async def _vector_search(self, query: str, k: int, doc_type: Optional[str], regulation_type: Optional[str]) -> List[Document]:
        """Internal vector search helper."""
        return await self._avector_only_search(query, k, doc_type, regulation_type)


    async def _avector_only_search(self, query: str, k: int, doc_type: Optional[str], regulation_type: Optional[str]) -> List[Document]:
        """Pure vector search with metadata filtering."""
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
            
            
            # Falling back to basic similarity search
            if "Error finding id" in error_msg or "Internal error" in error_msg:
                try:
                    return await asyncio.to_thread(self.vectorstore.similarity_search, query, **vector_kwargs)
                except Exception:
                    pass
            
            return []

    def _parse_date(self, date_str: Any) -> float:
        """Parse Vietnamese date strings to timestamp for comparison."""
        if not date_str:
            return 0.0
        
        date_str = str(date_str).strip()
        if date_str in self._date_cache:
            return self._date_cache[date_str]
        
        try:
            if len(date_str) == 4 and date_str.isdigit():
                t = datetime.strptime(date_str, "%Y").timestamp()
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
        """Keep only the NEWEST document per topic group."""
        def _normalize_title(title: str) -> str:
            t = title.lower().strip()
            t = unicodedata.normalize('NFKD', t)
            t = "".join([c for c in t if not unicodedata.combining(c)])
            
            if 'hoi dong hoc vu' in t or 'hdhv' in t: return 'hoi dong hoc vu'
            if 'quy che dao tao' in t: return 'quy che dao tao'
                
            t = re.sub(r'^\s*[\d\.\-\_]+', '', t)
            t = t.replace('dh&sdh', 'dai hoc va sau dai hoc').replace('dh sdh', 'dai hoc va sau dai hoc')
            t = t.replace('tb', 'thong bao').replace('kl', 'ket luan')
            t = t.replace('.md', '').replace('signed', '').replace('phien ban hop nhat', '')
            
            t = re.sub(r'[^a-z0-9\s]', ' ', t)
            return re.sub(r'\s+', ' ', t).strip()

        groups_best_ts: Dict[str, float] = {}
        for doc in docs:
            title = doc.metadata.get("title", doc.metadata.get("file_path", ""))
            key = _normalize_title(title)
            ts = self._parse_date(doc.metadata.get("issue_date"))
            groups_best_ts[key] = max(groups_best_ts.get(key, 0.0), ts)

        resolved: List[Document] = []
        for doc in docs:
            title = doc.metadata.get("title", doc.metadata.get("file_path", ""))
            key = _normalize_title(title)
            ts = self._parse_date(doc.metadata.get("issue_date"))
            if ts == groups_best_ts[key]:
                resolved.append(doc)
        return resolved
