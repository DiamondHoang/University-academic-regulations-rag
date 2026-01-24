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
    """Combines BM25 and vector search for better retrieval"""
    
    def __init__(self, vectorstore: Chroma, use_hybrid: bool = True):
        """Initialize retriever
        
        Args:
            vectorstore: Chroma vector store instance
            use_hybrid: Whether to use hybrid search
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
        """Retrieve documents using hybrid or vector search
        
        Args:
            query: Search query
            k: Number of results to retrieve
            doc_type: Filter by document type
            regulation_type: Filter by regulation type
            
        Returns:
            List of retrieved documents
        """
        try:
            if self.use_hybrid and self.bm25_retriever:
                retrieved_docs = self._hybrid_search(query, k)
            else:
                retrieved_docs = self.vectorstore.similarity_search(query, k=k*2)
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []
        
        # Apply filters
        retrieved_docs = self._apply_filters(
            retrieved_docs, doc_type, regulation_type, 
        )
        
        # Re-rank using cross-encoder if enabled
        if self.cross_encoder and retrieved_docs:
            retrieved_docs = self._rerank_by_cross_encoder(query, retrieved_docs)
        
        return retrieved_docs[:k]

    def _hybrid_search(self, query: str, k: int) -> List[Document]:
        """Combine BM25 and vector search using Reciprocal Rank Fusion (RRF)
        
        Args:
            query: Search query
            k: Number of results per retriever
            
        Returns:
            Combined ranked results
        """
        bm25_docs = self.bm25_retriever.invoke(query)
        vector_docs = self.vectorstore.similarity_search(query, k=k*2)

        rrf_k = Config.RRF_K
        doc_scores = {}

        def add_doc(doc: Document, rank: int) -> None:
            """Add document score using RRF formula"""
            score = 1 / (rrf_k + rank)
            doc_id = self._get_doc_id(doc, rank)
            
            if doc_id in doc_scores:
                doc_scores[doc_id] = (doc, doc_scores[doc_id][1] + score)
            else:
                doc_scores[doc_id] = (doc, score)

        for rank, doc in enumerate(bm25_docs[:k], 1):
            add_doc(doc, rank)

        for rank, doc in enumerate(vector_docs[:k], 1):
            add_doc(doc, rank)

        sorted_docs = sorted(doc_scores.values(), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in sorted_docs]

    
    def _get_doc_id(self, doc: Document, index: int) -> str:
        """Generate unique document ID
        
        Args:
            doc: Document
            index: Index for fallback
            
        Returns:
            Unique identifier
        """
        file_path = doc.metadata.get('file_path', '')
        chunk_id = doc.metadata.get('chunk_id', index)
        return f"{file_path}_{chunk_id}"
    
    def _apply_filters(
        self,
        docs: List[Document],
        doc_type: Optional[str],
        regulation_type: Optional[str],
    ) -> List[Document]:
        """Filter documents by metadata
        
        Args:
            docs: Documents to filter
            doc_type: Document type filter
            regulation_type: Regulation type filter

        Returns:
            Filtered documents
        """
        filtered = []
        for doc in docs:
            metadata = doc.metadata
            
            if doc_type and metadata.get("doc_type") != doc_type:
                continue
            if regulation_type and metadata.get("regulation_type") != regulation_type:
                continue
                        
            filtered.append(doc)
        
        return filtered
    
    def rank_by_recency(self, docs: List[Document]) -> List[Document]:
        """Sort documents by issue date (newest first)
        
        Args:
            docs: Documents to rank
            
        Returns:
            Ranked documents
        """
        def get_timestamp(doc: Document) -> float:
            """Extract timestamp from document metadata"""
            issue_date = doc.metadata.get("issue_date")
            if issue_date:
                try:
                    return datetime.strptime(issue_date, "%Y-%m-%d").timestamp()
                except (ValueError, TypeError):
                    return 0
            return 0
        
        return sorted(docs, key=get_timestamp, reverse=True)
    
    def _rerank_by_cross_encoder(self, query: str, docs: List[Document]) -> List[Document]:
        """Re-rank documents using cross-encoder for better relevance
        
        Args:
            query: Search query
            docs: Documents to re-rank
            
        Returns:
            Re-ranked documents sorted by cross-encoder score
        """
        if not docs:
            return docs
        
        try:
            # Prepare pairs for cross-encoder
            pairs = [[query, doc.page_content[:500]] for doc in docs]  # Use first 500 chars
            
            # Get relevance scores
            scores = self.cross_encoder.predict(pairs)
            
            # Sort by score
            ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
            return [doc for doc, _ in ranked]
        except Exception as e:
            logger.error(f"Cross-encoder re-ranking failed: {e}")
            return docs