import asyncio
import os
import shutil
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from config import Config
from uni_rag import UniversityRAG
from loader.doc_loader import RegulationDocumentLoader
from langchain_core.documents import Document

async def migrate_to_contextual():
    print("[Reindex] Starting migration to Contextual Retrieval...")
    
    # 1. Setup paths
    db_path = Config.DB_PATH
    backup_path = db_path + "_backup"
    # 2. Check if we have source files or existing DB
    embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
    loader = RegulationDocumentLoader(llm=UniversityRAG(embeddings=embeddings).response_generator._get_llm())
    
    documents = []
    if not os.path.exists(db_path):
        print(f"[Reindex] DB path {db_path} missing. Loading fresh from source files ({Config.BASE_PATH})...")
        documents = loader.load_documents()
    else:
        # Backup existing DB
        if os.path.exists(backup_path):
            shutil.rmtree(backup_path)
        shutil.copytree(db_path, backup_path)
        print(f"[Reindex] Backup created at {backup_path}")

        # Load documents from current DB
        db = Chroma(persist_directory=db_path, embedding_function=embeddings)
        print("[Reindex] Fetching documents from existing ChromaDB...")
        all_data = db.get()
        documents = [
            Document(page_content=all_data['documents'][i], metadata=all_data['metadatas'][i])
            for i in range(len(all_data['documents']))
        ]
        # Release locks
        del db
        import gc
        gc.collect()

    if not documents:
        print("[Error] No documents found in database or source files.")
        return
        
    print(f"[Reindex] Processing {len(documents)} source documents...")

    # 4. Contextualize & Hierarchical Splitting
    rag = UniversityRAG(embeddings=embeddings)
    loader = RegulationDocumentLoader(llm=rag.response_generator._get_llm())
    
    print("[Reindex] Creating Hierarchical Index (Pivots & Children)...")
    new_docs = await loader.load_hierarchical_documents()
    
    # 5. Update DB
    print(f"[Reindex] Updating ChromaDB with {len(new_docs)} hierarchical documents...")
    
    final_db_path = db_path
    if os.path.exists(db_path):
        try:
            import time
            time.sleep(1) # Wait a bit for OS to release locks
            shutil.rmtree(db_path)
            print("[Reindex] Old database deleted.")
        except PermissionError:
            final_db_path = db_path + "_new_" + str(int(time.time()))
            print(f"[Warning] 'vector_db' is locked. Saving to NEW directory: {final_db_path}")
            print("[Action Required] After this script finishes, please manually delete 'vector_db' and rename this new folder to 'vector_db'.")
    
    # Rebuild
    print(f"[Reindex] Re-building vectorstore at {final_db_path}")
    db = Chroma.from_documents(
        documents=new_docs,
        embedding=embeddings,
        persist_directory=final_db_path,
        collection_metadata=Config.EMBEDDING_KWARGS
    )

    print(f"[Reindex] Success! Database built at: {final_db_path}")
    if final_db_path != db_path:
        print(f"!!! TRANSLATION REQUIRED: Rename '{final_db_path}' to '{db_path}' manually.")
    
    print("[Reindex] Success! Contextual Retrieval migration complete.")
    print("[Reindex] You can now test the chat with higher accuracy and lower latency.")

if __name__ == "__main__":
    asyncio.run(migrate_to_contextual())
