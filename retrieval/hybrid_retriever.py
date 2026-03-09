import logging
import math
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder
from config import Config

logger = logging.getLogger(__name__)

class HybridRetriever:
    """Enforces a 3-stage retrieval: BM25 + Vector -> RRF Fusion -> Cross-Encoder Rerank -> Conflict Resolution."""

    def __init__(self, vectorstore: Chroma):
        """Initialize retriever

        Args:
            vectorstore: Chroma vector store instance
        """
        self.vectorstore = vectorstore
        self.bm25_retriever: Optional[BM25Retriever] = None
        self.bm25_built = False
        self._date_cache: Dict[str, float] = {}  # Cache for parsed timestamps

        # Enforce cross-encoder for re-ranking
        try:
            self.cross_encoder = CrossEncoder(Config.CROSS_ENCODER_MODEL)
        except Exception as e:
            logger.error(f"Failed to load cross-encoder: {e}")
            self.cross_encoder = None

    def build_bm25(self, chunks: List[Document]) -> None:
        """Build BM25 retriever from document chunks"""
        if not self.bm25_built:
            try:
                self.bm25_retriever = BM25Retriever.from_documents(chunks)
                self.bm25_retriever.k = Config.BM25_K
                self.bm25_built = True
            except Exception as e:
                logger.error(f"Failed to build BM25 retriever: {e}")
                self.bm25_retriever = None

    def retrieve(
        self,
        query: str,
        k: int = 5,
        doc_type: Optional[str] = None,
        regulation_type: Optional[str] = None,
    ) -> List[Document]:
        """Retrieve documents using simplified pipeline."""
        try:
            # 1. Search Stage
            if not Config.USE_HYBRID_SEARCH:
                retrieved_docs = self._vector_only_search(query, k, doc_type, regulation_type)
            else:
                retrieved_docs = self._hybrid_search(query, k, doc_type, regulation_type)
            
            if not retrieved_docs and (doc_type or regulation_type):
                logger.warning(f"Filtered retrieval returned 0 results. Falling back to unfiltered.")
                if not Config.USE_HYBRID_SEARCH:
                    retrieved_docs = self._vector_only_search(query, k, None, None)
                else:
                    retrieved_docs = self._hybrid_search(query, k, None, None)

            # 2. Post-processing
            if Config.USE_PARENT_CHILD:
                retrieved_docs = self._reconstruct_parents_and_deduplicate(retrieved_docs)
            else:
                retrieved_docs = self._simple_deduplicate(retrieved_docs)

            # 3. Conflict resolution: keep only the NEWEST doc per regulation topic
            retrieved_docs = self._resolve_conflicts_by_date(retrieved_docs)

            return retrieved_docs[:k]
        except Exception as e:
            logger.error(f"Retrieval pipeline failed: {e}")
            return []

    def _apply_recency_boost(self, docs: List[Document]) -> List[Document]:
        """Calculate a combined score: (1 - W) * relevance + W * recency."""
        if not docs:
            return docs

        # Calculate recency timestamps
        timestamps = [self._parse_date(doc.metadata.get("issue_date")) for doc in docs]
        min_ts = min(timestamps)
        max_ts = max(timestamps)
        ts_range = max_ts - min_ts if max_ts > min_ts else 1.0

        weight = getattr(Config, "RECENCY_WEIGHT", 0.0)
        
        scored_docs = []
        for doc, ts in zip(docs, timestamps):
            # Normalize relevance (assuming 0-1 from vector search or rrf)
            relevance = doc.metadata.get("confidence_score", doc.metadata.get("rrf_score", 0.5))
            
            # Normalize recency to [0, 1]
            recency = (ts - min_ts) / ts_range
            
            # Combined score
            final_score = (1 - weight) * relevance + weight * recency
            doc.metadata["final_combined_score"] = final_score
            doc.metadata["confidence_score"] = final_score # Update for downstream filtering
            scored_docs.append((doc, final_score))

        # Sort by final score
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs]

    def _vector_only_search(self, query: str, k: int, doc_type: Optional[str], regulation_type: Optional[str]) -> List[Document]:
        """Pure vector search with metadata filtering, using real relevance scores."""
        candidate_k = max(k * 4, 20)
        
        chroma_filter = {}
        if doc_type:
            chroma_filter["doc_type"] = doc_type
        if regulation_type:
            chroma_filter["regulation_type"] = regulation_type

        vector_kwargs = {"k": candidate_k}
        if chroma_filter:
            if len(chroma_filter) == 1:
                key, val = list(chroma_filter.items())[0]
                vector_kwargs["filter"] = {key: val}
            else:
                vector_kwargs["filter"] = {"$and": [{k: v} for k, v in chroma_filter.items()]}

        # Use relevance scores so CONFIDENCE_THRESHOLD can actually filter weak hits
        try:
            scored = self.vectorstore.similarity_search_with_relevance_scores(query, **vector_kwargs)
            docs = []
            for doc, score in scored:
                doc.metadata["confidence_score"] = round(float(score), 4)
                docs.append(doc)
        except Exception:
            # Fallback to plain similarity_search if scored variant not available
            docs = self.vectorstore.similarity_search(query, **vector_kwargs)
            for doc in docs:
                doc.metadata["confidence_score"] = 0.5
        return docs

    def _simple_deduplicate(self, docs: List[Document]) -> List[Document]:
        """Simple deduplication by file_path and title."""
        deduped = []
        seen_files = {}
        max_per_file = getattr(Config, "MAX_CHUNKS_PER_FILE", 1)
        
        for doc in docs:
            file_path = doc.metadata.get("file_path", "unknown")
            count = seen_files.get(file_path, 0)
            if count < max_per_file:
                deduped.append(doc)
                seen_files[file_path] = count + 1
        return deduped

    def _hybrid_search(self, query: str, k: int, doc_type: Optional[str], regulation_type: Optional[str]) -> List[Document]:
        """Combine BM25 and vector search using RRF with large candidate pool and pre-filtering."""
        candidate_k = max(k * 10, 50)
        
        # Build Chroma filter
        chroma_filter = {}
        if doc_type:
            chroma_filter["doc_type"] = {"$eq": doc_type}
        if regulation_type:
            chroma_filter["regulation_type"] = {"$eq": regulation_type}

        # Vector search with metadata filtering
        vector_kwargs: Dict[str, Any] = {"k": candidate_k}
        if chroma_filter:
            if len(chroma_filter) == 1:
                key, val = list(chroma_filter.items())[0]
                vector_kwargs["filter"] = {key: val["$eq"]}
            else:
                vector_kwargs["filter"] = {"$and": [{k: v["$eq"]} for k, v in chroma_filter.items()]}
        
        vector_docs = self.vectorstore.similarity_search(query, **vector_kwargs)

        # BM25 search with metadata filtering
        bm25_docs = []
        if self.bm25_retriever:
            try:
                # Retrieve slightly more for BM25 to account for post-filtering drops
                bm25_candidates = self.bm25_retriever.invoke(query)[:candidate_k * 3]
                bm25_docs = self._apply_filters(bm25_candidates, doc_type, regulation_type)[:candidate_k]
            except Exception:
                bm25_docs = []

        rrf_k = Config.RRF_K
        doc_scores = {}

        def add_doc(doc: Document, rank: int) -> None:
            """Add document score using RRF formula and deduplicate by id."""
            score = 1 / (rrf_k + rank)
            doc_id = self._get_doc_id(doc, rank)

            if doc_id in doc_scores:
                doc_obj, old_score = doc_scores[doc_id]
                doc_scores[doc_id] = (doc_obj, old_score + score)
            else:
                doc.metadata["rrf_score"] = score
                doc_scores[doc_id] = (doc, score)

        for rank, doc in enumerate(bm25_docs, 1):
            add_doc(doc, rank)

        for rank, doc in enumerate(vector_docs, 1):
            add_doc(doc, rank)

        sorted_docs = sorted(doc_scores.values(), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in sorted_docs]

    def _get_doc_id(self, doc: Document, index: int) -> str:
        """Generate unique document ID from metadata."""
        file_path = doc.metadata.get("file_path", "")
        chunk_id = doc.metadata.get("chunk_id", index)
        return f"{file_path}_{chunk_id}"

    def _apply_filters(
        self,
        docs: List[Document],
        doc_type: Optional[str],
        regulation_type: Optional[str],
    ) -> List[Document]:
        """Filter documents by metadata."""
        filtered = []
        for doc in docs:
            metadata = doc.metadata
            if doc_type and metadata.get("doc_type") != doc_type:
                continue
            if regulation_type and metadata.get("regulation_type") != regulation_type:
                continue
            filtered.append(doc)
        return filtered

    # def _rerank_by_cross_encoder(self, query: str, docs: List[Document]) -> List[Document]:
    #     """Re-rank documents combining normalized relevance + normalized recency + priority boost.
        
    #     Both relevance and recency are independently normalized to [0,1] so neither
    #     dominates by default. The weights are configurable via Config.RECENCY_WEIGHT.
    #     """
    #     if not docs: return docs

    #     try:
    #         texts = [doc.page_content[:512] for doc in docs]
    #         pairs = [(query, t) for t in texts]
    #         raw_scores = self.cross_encoder.predict(pairs)

    #         # --- Step 1: Normalize relevance to [0, 1] ---
    #         min_rel = float(min(raw_scores))
    #         max_rel = float(max(raw_scores))
    #         rel_diff = max_rel - min_rel if max_rel > min_rel else 1.0

    #         # --- Step 2: Parse recency timestamps for each doc ---
    #         recency_ts = []
    #         for doc in docs:
    #             issue_date = doc.metadata.get("issue_date")
    #             t = 0.0
    #             if issue_date:
    #                 if issue_date in self._date_cache:
    #                     t = self._date_cache[issue_date]
    #                 else:
    #                     try:
    #                         date_str = str(issue_date).replace("/", "-")
    #                         if len(date_str) == 4:
    #                             t = datetime.strptime(date_str, "%Y").timestamp()
    #                         elif "-" in date_str:
    #                             parts = date_str.split("-")
    #                             if len(parts[0]) == 4:  # YYYY-MM-DD
    #                                 t = datetime.strptime(date_str, "%Y-%m-%d").timestamp()
    #                             else:  # DD-MM-YYYY
    #                                 t = datetime.strptime(date_str, "%d-%m-%Y").timestamp()
    #                         else:
    #                             t = datetime.strptime(date_str, "%d-%m-%Y").timestamp()
    #                         self._date_cache[issue_date] = t
    #                     except Exception:
    #                         t = 0.0
    #             recency_ts.append(t)

    #         # Normalize recency to [0, 1] using min-max across candidates
    #         min_rec = min(recency_ts)
    #         max_rec = max(recency_ts)
    #         rec_diff = max_rec - min_rec if max_rec > min_rec else 1.0

    #         # --- Step 3: Combine with configurable weights ---
    #         RELEVANCE_WEIGHT = 1.0 - Config.RECENCY_WEIGHT  # e.g., 0.5
    #         final_scores = []
    #         for doc, raw_score, ts in zip(docs, raw_scores, recency_ts):
    #             norm_relevance = (float(raw_score) - min_rel) / rel_diff
    #             norm_recency = (ts - min_rec) / rec_diff
    #             priority_boost = Config.PRIORITY_BOOST if doc.metadata.get("priority") == "high" else 0.0

    #             final = (RELEVANCE_WEIGHT * norm_relevance
    #                      + Config.RECENCY_WEIGHT * norm_recency
    #                      + priority_boost)

    #             doc.metadata["confidence_score"] = round(norm_relevance, 3)
    #             doc.metadata["recency_score"] = round(norm_recency, 3)
    #             final_scores.append((doc, final))

    #         ranked = sorted(final_scores, key=lambda x: x[1], reverse=True)
    #         return [doc for doc, _ in ranked]

    #     except Exception as e:
    #         logger.error(f"Reranking failed: {e}")
    #         for doc in docs:
    #             doc.metadata["confidence_score"] = 0.5
    #         return docs

    def _rerank_by_cross_encoder(self, query: str, docs: List[Document]) -> List[Document]:
        """Re-rank documents using pure absolute relevance from the Cross-Encoder."""
        if not docs:
            return docs

        try:
            texts = [doc.page_content[:512] for doc in docs]
            pairs = [(query, t) for t in texts]
            raw_scores = self.cross_encoder.predict(pairs)

            # --- Simple Min-Max Normalization to [0, 1] for UI display ---
            min_score = float(min(raw_scores))
            max_score = float(max(raw_scores))
            score_range = max_score - min_score if max_score > min_score else 1.0

            final_scores = []
            for doc, raw_score in zip(docs, raw_scores):
                # Calculate simple normalized relevance score
                normalized_relevance = (float(raw_score) - min_score) / score_range
                
                # Assign scores for UI and downstream filtering
                doc.metadata["confidence_score"] = round(normalized_relevance, 3)
                doc.metadata["weighted_score"] = round(normalized_relevance, 3) # Keep for backward compatibility

                final_scores.append((doc, normalized_relevance))

            # Rank purely by relevance score
            ranked = sorted(final_scores, key=lambda x: x[1], reverse=True)
            return [doc for doc, _ in ranked]

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            for doc in docs:
                doc.metadata["confidence_score"] = 0.5
            return docs
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
        """Resolve regulation conflicts by keeping only the NEWEST document per topic group.

        Groups documents by a normalized title key (lowercase, strip numbers/dates from end).
        Within each group, keeps the one with the latest issue_date.
        This ensures the LLM never receives outdated conflicting regulations.
        """
        import re as _re

        def _normalize_title(title: str) -> str:
            # Strip trailing year/number/date suffixes for grouping
            t = title.lower().strip()
            # Remove trailing: digits, slashes, hyphens, year patterns
            t = _re.sub(r'[\s\-/]*(\d{4}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|số\s*\d+[\s\S]{0,10})\s*$', '', t)
            return t.strip()

        groups: Dict[str, Tuple[Document, float]] = {}  # key -> (best_doc, best_ts)

        for doc in docs:
            title = doc.metadata.get("title", doc.metadata.get("file_path", ""))
            key = _normalize_title(title)
            ts = self._parse_date(doc.metadata.get("issue_date"))

            if key not in groups:
                groups[key] = (doc, ts)
            else:
                _, best_ts = groups[key]
                if ts > best_ts:
                    groups[key] = (doc, ts)
                    logger.debug(f"Conflict resolved: kept newer doc '{title}' (ts={ts}) over older (ts={best_ts})")

        # Preserve original rank order (first occurrence of each group wins position)
        seen_keys: set = set()
        resolved: List[Document] = []
        for doc in docs:
            title = doc.metadata.get("title", doc.metadata.get("file_path", ""))
            key = _normalize_title(title)
            if key not in seen_keys:
                seen_keys.add(key)
                best_doc, _ = groups[key]
                resolved.append(best_doc)
        return resolved

    def _reconstruct_parents_and_deduplicate(self, docs: List[Document]) -> List[Document]:
        """Reconstruct parent chunks and deduplicate, prioritizing the LATEST documents.
        
        If multiple chunks refer to the same content/regulation, the one with the 
        newest issue_date is kept.
        """
        reconstructed = []
        seen_parents = set()
        file_counts: Dict[str, int] = {}
        max_per_file = getattr(Config, "MAX_CHUNKS_PER_FILE", 1)

        for doc in docs:
            metadata = doc.metadata.copy()
            parent_id = metadata.get("parent_id")
            parent_content = metadata.get("parent_content")
            
            if parent_id and parent_content:
                if parent_id in seen_parents:
                    continue
                seen_parents.add(parent_id)
                page_content = parent_content
                metadata.pop("parent_content", None)
            else:
                page_content = doc.page_content

            file_path = metadata.get("file_path", "")
            current_count = file_counts.get(file_path, 0)
            
            if current_count < max_per_file:
                file_counts[file_path] = current_count + 1
                reconstructed.append(Document(page_content=page_content, metadata=metadata))
        
        # FINAL STEP: Strictly sort the final results by Date descending
        reconstructed.sort(key=lambda d: self._parse_date(d.metadata.get("issue_date")), reverse=True)
                
        return reconstructed
