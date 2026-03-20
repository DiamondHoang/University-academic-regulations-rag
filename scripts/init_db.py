import os
import sys

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from loader.doc_loader import RegulationDocumentLoader
from uni_rag import UniversityRAG



def init_database():
    """Build or verify the vector database before web server starts."""
    try:
        db_path = Config.DB_PATH
        
        # Ensure the parent directory exists
        db_parent = os.path.dirname(db_path)
        if db_parent:
            os.makedirs(db_parent, exist_ok=True)
            
        loader = RegulationDocumentLoader(base_path=Config.BASE_PATH)
        documents = loader.load_documents()
        
        if not documents:
            return False
            
        rag = UniversityRAG()
        
        # The build_vectorstore method handles both creation and loading
        # Passing force_rebuild=False will use existing DB if it looks valid
        rag.build_vectorstore(documents, force_rebuild=False)
        
    except Exception:
        return False

if __name__ == "__main__":
    success = init_database()
    if not success:
        sys.exit(1)
    sys.exit(0)
