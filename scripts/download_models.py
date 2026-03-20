from sentence_transformers import CrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings
from config import Config

def download():
    print("Downloading Embedding model...")
    HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
    
    print("Downloading Reranker model...")
    try:
        if Config.USE_RERANKER:
            CrossEncoder(Config.RERANKER_MODEL)
    except Exception as e:
        print(f"Error downloading reranker: {e}")

if __name__ == "__main__":
    download()
