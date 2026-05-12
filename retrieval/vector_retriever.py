import asyncio
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder
from config import Config

class VectorRetriever:
    """Retrieves relevant academic regulation documents using vector search and re-ranking."""

    def __init__(self, vectorstore: Chroma):
        """Initialize the retriever with a valid ChromaDB instance.
        
        Args:
            vectorstore: The pre-loaded vector database.
        """
        self.vectorstore = vectorstore
        self._date_cache: Dict[str, float] = {}

        # 1. Initialize Re-ranker if enabled
        if Config.USE_RERANKER:
            try:
                print(f"[Retrieval] Loading re-ranker: {Config.RERANKER_MODEL}")
                self.reranker = CrossEncoder(Config.RERANKER_MODEL)
            except Exception as e:
                print(f"[Retrieval] Re-ranker failed to load: {e}. Falling back to cosine only.")
                self.reranker = None
        else:
            self.reranker = None

    async def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """Perform a multi-stage retrieval process.
        
        Process:
            1. Vector search to get initial candidates.
            2. Re-ranking (if enabled) to prioritize results.
            
        Args:
            query: Standing query to search for.
            k: Number of documents to return.
            
        Returns:
            Ranked list of LangChain Document objects.
        """
        try:
            # 1. Broad Vector Search
            docs = await self._vector_search(query, k)
            
            if not docs:
                return []

            # 2. Re-ranking
            if self.reranker and len(docs) > 1:
                # Skip reranking if we have Contextual Retrieval (Zero-latency retrieval)
                if Config.USE_CONTEXTUAL_RETRIEVAL:
                    print("[Retrieval] Contextual Retrieval is active. Skipping re-ranking for maximum speed.")
                    return docs[:k]

                try:
                    # Rerank first 700 chars of each doc for speed efficiency
                    pairs = [[query, doc.page_content[:700]] for doc in docs]
                    scores = await asyncio.to_thread(
                        self.reranker.predict,
                        pairs,
                        batch_size=16,
                        show_progress_bar=False,
                    )
                    
                    for i, score in enumerate(scores):
                        docs[i].metadata["rerank_score"] = float(score)
                        docs[i].metadata["confidence_score"] = round(float(score), 4)

                    docs.sort(key=lambda x: x.metadata["rerank_score"], reverse=True)
                    print(f"[Retrieval] Re-ranked {len(docs)} documents.")
                except Exception as e:
                    print(f"[Retrieval] Re-ranking error: {e}. Falling back to vector order.")

            return docs[:k]
        except Exception as e:
            print(f"[Retrieval] Fatal error in retrieve: {e}")
            return []

    async def _vector_search(self, query: str, k: int) -> List[Document]:
        """Execute the pure similarity search against the vectorDB."""
        try:
            scored = await asyncio.to_thread(
                self.vectorstore.similarity_search_with_relevance_scores,
                query,
                k=k,
            )
            
            docs = []
            for doc, score in scored:
                doc.metadata["confidence_score"] = round(float(score), 4)
                docs.append(doc)
            return docs
        except Exception as e:
            print(f"[Retrieval] Vector search failed: {e}")
            return []