from typing import List, Optional
from datetime import datetime
import logging

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder
from config import Config

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Combines BM25 and vector search with optional cross-encoder re-ranking."""

    def __init__(self, vectorstore: Chroma, use_hybrid: bool = True):
        """Initialize retriever

        Args:
            vectorstore: Chroma vector store instance
            use_hybrid: Whether to use hybrid search (BM25 + vector)
        """
        self.vectorstore = vectorstore
        self.use_hybrid = use_hybrid
        self.bm25_retriever: Optional[BM25Retriever] = None
        self.bm25_built = False

        # Initialize cross-encoder for re-ranking if enabled
        self.cross_encoder = None
        if Config.USE_CROSS_ENCODER_RERANK:
            try:
                self.cross_encoder = CrossEncoder(Config.CROSS_ENCODER_MODEL)
            except Exception as e:
                logger.error(f"Failed to load cross-encoder: {e}")

    def build_bm25(self, chunks: List[Document]) -> None:
        """Build BM25 retriever from document chunks

        Args:
            chunks: Document chunks to index
        """
        if self.use_hybrid and not self.bm25_built:
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
        """Retrieve documents using hybrid or vector search with confidence scores

        The returned list is already filtered and re-ranked (if cross-encoder
        is enabled). Documents have confidence scores attached in metadata.

        Args:
            query: Search query
            k: Number of results to retrieve
            doc_type: Filter by document type
            regulation_type: Filter by regulation type

        Returns:
            List of retrieved documents with confidence scores in metadata
        """
        try:
            if self.use_hybrid and self.bm25_retriever:
                retrieved_docs = self._hybrid_search(query, k)
            else:
                retrieved_docs = self.vectorstore.similarity_search(query, k=k * 2)
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []

        # Apply filters
        retrieved_docs = self._apply_filters(retrieved_docs, doc_type, regulation_type)

        # Re-rank using cross-encoder if enabled (also returns scores)
        if self.cross_encoder and retrieved_docs:
            retrieved_docs = self._rerank_by_cross_encoder(query, retrieved_docs)

        return retrieved_docs[:k]

    def _hybrid_search(self, query: str, k: int) -> List[Document]:
        """Combine BM25 and vector search using Reciprocal Rank Fusion (RRF).
        
        Attaches RRF scores to document metadata for later filtering.
        """
        bm25_docs = []
        if self.bm25_retriever:
            try:
                bm25_docs = self.bm25_retriever.invoke(query)[: k * 2]
            except Exception:
                bm25_docs = []

        vector_docs = self.vectorstore.similarity_search(query, k=k * 2)

        rrf_k = Config.RRF_K
        doc_scores = {}

        def add_doc(doc: Document, rank: int) -> None:
            """Add document score using RRF formula and deduplicate by id."""
            score = 1 / (rrf_k + rank)
            doc_id = self._get_doc_id(doc, rank)

            if doc_id in doc_scores:
                doc, old_score = doc_scores[doc_id]
                doc_scores[doc_id] = (doc, old_score + score)
            else:
                # Attach RRF score to metadata
                doc.metadata["rrf_score"] = score
                doc_scores[doc_id] = (doc, score)

        for rank, doc in enumerate(bm25_docs[:k], 1):
            add_doc(doc, rank)

        for rank, doc in enumerate(vector_docs[:k], 1):
            add_doc(doc, rank)

        # Update metadata with final RRF scores
        sorted_docs = sorted(doc_scores.values(), key=lambda x: x[1], reverse=True)
        for doc, score in sorted_docs:
            doc.metadata["rrf_score"] = score
        
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

    # NOTE: recency-only ranking function removed to avoid accidental
    # double-sorting that would discard cross-encoder relevance ordering.
    # `_rerank_by_cross_encoder` already mixes relevance with a recency signal
    # using `Config.RECENCY_WEIGHT`. If strict recency ordering is required,
    # callers should perform it explicitly.

    def _rerank_by_cross_encoder(self, query: str, docs: List[Document]) -> List[Document]:
        """Re-rank documents using a cross-encoder and a recency signal.
        
        Attaches normalized confidence scores (0-1) to document metadata.
        """
        if not docs:
            return docs

        try:
            pairs = [[query, doc.page_content[:500]] for doc in docs]
            scores = self.cross_encoder.predict(pairs)

            now = datetime.now().timestamp()
            final_scores = []

            for doc, score in zip(docs, scores):
                issue_date = doc.metadata.get("issue_date")
                if issue_date:
                    try:
                        t = datetime.strptime(issue_date, "%Y-%m-%d").timestamp()
                        recency = 1 / (1 + (now - t) / (365 * 24 * 3600))
                    except Exception:
                        recency = 0
                else:
                    recency = 0

                final = float(score) + Config.RECENCY_WEIGHT * recency
                final_scores.append(final)

            # Normalize scores to 0-1 range
            if final_scores:
                max_score = max(final_scores)
                min_score = min(final_scores)
                score_range = max_score - min_score if max_score > min_score else 1.0
                
                normalized_scores = []
                for doc, final_score in zip(docs, final_scores):
                    normalized = (final_score - min_score) / score_range
                    normalized = max(0.0, min(1.0, normalized))
                    doc.metadata["confidence_score"] = float(normalized)
                    normalized_scores.append((doc, normalized))
            else:
                for doc in docs:
                    doc.metadata["confidence_score"] = 0.0
                normalized_scores = [(doc, 0.0) for doc in docs]

            ranked = sorted(normalized_scores, key=lambda x: x[1], reverse=True)
            return [doc for doc, _ in ranked]

        except Exception as e:
            logger.error(f"Cross-encoder re-ranking failed: {e}")
            # Fallback: assign equal confidence
            for doc in docs:
                doc.metadata["confidence_score"] = 0.5
            return docs
