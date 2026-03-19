"""CLI Entry point for the University Academic Regulations RAG system"""
import asyncio
import logging
from config import Config
from loader.doc_loader import RegulationDocumentLoader
from uni_rag import UniversityRAG

# Standardized logging for CLI
logging.basicConfig(level=logging.ERROR, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def run_chat():
    """Main execution flow for interactive chat"""
    print("=" * 70)
    print("CHATBOT QUY ĐỊNH HỌC VỤ")
    print("=" * 70)
    
    # 1. Initialize System
    print("\n[status] Đang tải dữ liệu và khởi tạo hệ thống...")
    loader = RegulationDocumentLoader(base_path=Config.BASE_PATH)
    rag = UniversityRAG()
    
    documents = loader.load_documents()
    if not documents:
        print("[error] Không tìm thấy tài liệu nào trong thư mục 'md'.")
        return

    # Build vectorstore (handles persistence internally)
    rag.build_vectorstore(documents, force_rebuild=False)
    
    # 2. Start Chat Loop
    print("\n[status] Hệ thống đã sẵn sàng.")
    print("\nCác lệnh có sẵn:")
    print("  - 'exit'/'quit': Thoát ứng dụng")
    print("  - 'clear': Xóa lịch sử hội thoại")
    print("  - 'history': Xem lịch sử hội thoại")
    
    current_filter = {"doc_type": None, "regulation_type": None}
    
    while True:
        try:
            user_input = input("\nBạn: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ["exit", "quit"]:
                print("\nCảm ơn bạn đã sử dụng chatbot!")
                break
            
            if user_input.lower() == "clear":
                rag.memory.clear()
                print("Đã xóa lịch sử hội thoại.")
                continue
            
            if user_input.lower() == "history":
                show_history(rag)
                continue
            
            # Process query
            answer = await rag.aquery(
                user_input,
                doc_type=current_filter["doc_type"],
                regulation_type=current_filter["regulation_type"]
            )
            print(f"\nBot:\n{answer}")
            
        except KeyboardInterrupt:
            print("\n\nĐã dừng chatbot.")
            break
        except Exception as e:
            logger.error(f"Error in chat loop: {e}", exc_info=True)
            print(f"\n[Lỗi] Có sự cố xảy ra: {e}. Vui lòng thử lại.")

def show_history(rag: UniversityRAG) -> None:
    """Display conversation history summary"""
    if not rag.memory.history:
        print("Chưa có lịch sử hội thoại.")
        return
        
    print("\n" + "=" * 60)
    print("LỊCH SỬ HỘI THOẠI")
    print("=" * 60)
    for i, turn in enumerate(rag.memory.history, 1):
        print(f"\n{i}. Câu hỏi: {turn['question']}")
        preview = turn['answer'][:200] + "..." if len(turn['answer']) > 200 else turn['answer']
        print(f"   Trả lời: {preview}")
    print("=" * 60)

if __name__ == "__main__":
    try:
        asyncio.run(run_chat())
    except KeyboardInterrupt:
        pass
