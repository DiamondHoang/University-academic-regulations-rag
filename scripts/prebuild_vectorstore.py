import os
import sys
import asyncio

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from loader.doc_loader import RegulationDocumentLoader
from uni_rag import UniversityRAG
from langchain_huggingface import HuggingFaceEmbeddings

async def prebuild():
    """Pre-build script: builds the vector store before the server starts.

    Run via entrypoint.sh or manually:
        python scripts/prebuild_vectorstore.py

    Pipeline:
        1. Load markdown documents.
        2. Initialize the embedding model.
        3. Build and persist the vector store to disk.
    """
    print("Starting vectorstore pre-build...")

    # Step 1: Load documents from the configured directory
    loader = RegulationDocumentLoader(base_path=Config.BASE_PATH)
    print(f"Loading documents from {Config.BASE_PATH}...")
    documents = loader.load_documents()
    print(f"Loaded {len(documents)} documents.")

    if not documents:
        print("No documents found. Skipping vectorstore build.")
        return

    # Step 2: Initialize embeddings and RAG (response generator not needed for pre-build)
    config = Config.as_dict()
    embeddings = HuggingFaceEmbeddings(model_name=config["embedding_model"])

    rag = UniversityRAG(embeddings=embeddings)

    # Step 3: Fully rebuild the vector store from scratch (force_rebuild=True)
    db_path = config["db_path"]
    print(f"Building vectorstore at {db_path}...")
    await asyncio.to_thread(rag.build_vectorstore, documents, force_rebuild=True)

    print("Vectorstore pre-build complete.")


if __name__ == "__main__":
    asyncio.run(prebuild())
