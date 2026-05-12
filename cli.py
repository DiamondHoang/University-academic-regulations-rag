import asyncio
from config import Config
from loader.doc_loader import RegulationDocumentLoader
from uni_rag import UniversityRAG



async def run_chat():
    """Interactive terminal chat loop.

    Special commands:
        exit / quit  -- exit the program
        clear        -- clear conversation history
        history      -- display the history of Q&A turns
    """
    
    # 1. Initialize System
    loader = RegulationDocumentLoader(base_path=Config.BASE_PATH)
    rag = UniversityRAG()
    
    documents = loader.load_documents()
    if not documents:
        return

    # Build vectorstore (handles persistence internally)
    await rag.build_vectorstore(documents, force_rebuild=False)
    
    
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
            
            import time
            start_time = time.perf_counter()
            
            answer = await rag.aquery(
                user_input
            )
            
            end_time = time.perf_counter()
            duration = end_time - start_time
            
            print(f"\nHệ thống: {answer}")
            print(f"[Thời gian phản hồi: {duration:.2f} giây]")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\nLỗi: {str(e)}")
            continue

def show_history(rag: UniversityRAG) -> None:
    """Print the Q&A history of the current session to the terminal."""
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
