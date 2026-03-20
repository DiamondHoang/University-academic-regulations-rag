import asyncio
from config import Config
from loader.doc_loader import RegulationDocumentLoader
from uni_rag import UniversityRAG



async def run_chat():
    """Main execution flow for interactive chat"""
    
    # 1. Initialize System
    # 1. Initialize System
    loader = RegulationDocumentLoader(base_path=Config.BASE_PATH)
    rag = UniversityRAG()
    
    documents = loader.load_documents()
    if not documents:
        return

    # Build vectorstore (handles persistence internally)
    rag.build_vectorstore(documents, force_rebuild=False)
    
    current_filter = {"doc_type": None, "regulation_type": None}
    
    while True:
        try:
            user_input = input("\nBạn: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ["exit", "quit"]:
                break
            
            if user_input.lower() == "clear":
                rag.memory.clear()
                continue
            
            if user_input.lower() == "history":
                show_history(rag)
                continue
            
            answer = await rag.aquery(
                user_input,
                doc_type=current_filter["doc_type"],
                regulation_type=current_filter["regulation_type"]
            )
            print(f"\nHệ thống: {answer}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\nLỗi: {str(e)}")
            continue

def show_history(rag: UniversityRAG) -> None:
    """Display conversation history summary"""
    if not rag.memory.history:
        print("\nLịch sử trống.")
        return
    for i, turn in enumerate(rag.memory.history, 1):
        print(f"\n[{i}] Bạn: {turn['question']}")
        print(f"[{i}] Hệ thống: {turn['answer']}")

if __name__ == "__main__":
    try:
        asyncio.run(run_chat())
    except KeyboardInterrupt:
        pass
