import os
import sys
import asyncio

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from loader.doc_loader import RegulationDocumentLoader
from uni_rag import UniversityRAG
from langchain_huggingface import HuggingFaceEmbeddings
from retrieval.response_generator import ResponseGenerator

async def prebuild():
    print("Starting vectorstore pre-build...")
    
    # 1. Load documents
    loader = RegulationDocumentLoader(base_path=Config.BASE_PATH)
    print(f"Loading documents from {Config.BASE_PATH}...")
    documents = loader.load_documents()
    print(f"Loaded {len(documents)} documents.")
    
    if not documents:
        print("No documents found. Skipping vectorstore build.")
        return

    # 2. Initialize components
    config = Config.as_dict()
    embeddings = HuggingFaceEmbeddings(model_name=config["embedding_model"])
    generator = ResponseGenerator(config)
    
    rag = UniversityRAG(
        embeddings=embeddings,
        response_generator=generator
    )
    
    # 3. Build vectorstore
    db_path = config["db_path"]
    print(f"Building vectorstore at {db_path}...")
    
    # Run in thread since Chroma build is CPU intensive
    await asyncio.to_thread(rag.build_vectorstore, documents, force_rebuild=True)
    
    print("Vectorstore pre-build complete.")

if __name__ == "__main__":
    asyncio.run(prebuild())
