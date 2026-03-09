import logging
import re
import textwrap
from typing import List, Dict, Tuple, Optional
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from config import Config

logger = logging.getLogger(__name__)

class ResponseGenerator:
    """Modular Response Generator with multi-source grounding and Vietnamese prompts."""

    def __init__(self, llm: ChatOllama):
        """Initialize response generator
        
        Args:
            llm: ChatOllama instance
        """
        self.llm = llm
        self.confidence_threshold = Config.CONFIDENCE_THRESHOLD

    def generate(
        self,
        query: str,
        documents: List[Document],
        conversation_history: str = "",
        analysis: Optional[Dict] = None,
        clean_mode: bool = False,
    ) -> Dict:
        """Generate response based on retrieved documents"""
        analysis = analysis or {}
        
        # 1. Selection & Thresholding
        selected_docs = self._filter_by_confidence(documents)
        
        if not selected_docs:
            answer = f"Không tìm thấy quy định về '{query}' trong hệ thống văn bản của nhà trường."
            return {"answer": answer, "confidence": 0.0, "sources": []}

        # Keep top K documents based on relevance — preserve retriever's ordering
        # (retriever already sorted by date descending via _resolve_conflicts_by_date)
        max_docs: int = getattr(Config, "MAX_RESPONSE_DOCS", 3)
        selected_docs = selected_docs[:max_docs]

        # 2. Context Building (Handles multiple sources and content with reordering)
        context, sources = self._build_context(selected_docs)

        # 3. LLM Call
        try:
            messages = self._build_messages(query, context, primary_source_index=1)
            
            logger.info(f"Generating answer with {len(selected_docs)} sources.")
            
            response = self.llm.invoke(messages)
            answer = response.content.strip() if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            answer = "Hệ thống gặp sự cố khi tạo câu trả lời."

        # 4. Output Processing & Formatting
        final_response = self._format_response(answer, sources)
        
        if clean_mode:
            return {"answer": answer, "confidence": 1.0, "sources": []}
            
        return {
            "answer": final_response,
            "confidence": selected_docs[0].metadata.get("weighted_score", selected_docs[0].metadata.get("confidence_score", 0.0)) if selected_docs else 0.0,
            "sources": sources,
        }

    def _format_response(self, answer: str, sources: List[Dict]) -> str:
        """Format final answer with sequential source indexing."""
        if not sources:
            return answer

        # 0. Pre-collapse: remove duplicate "[Nguồn N] [N]" → "[N]"
        #    LLM sometimes writes "[Nguồn 1] [1]" — collapse to just "[1]"
        answer = re.sub(r'\[Nguồn\s+(\d+)\]\s*\[\1\]', r'[\1]', answer)
        # Also collapse "[Nguồn N]" alone → "[N]" for consistency
        answer = re.sub(r'\[Nguồn\s+(\d+)\]', r'[\1]', answer)

        # 1. Remap inline citations [N] to sequential indices
        cite_pattern = r'\[(\d+)\]'
        all_matches = list(re.finditer(cite_pattern, answer))

        orig_to_new: Dict[int, Dict] = {}
        next_new_idx: int = 1

        for m in all_matches:
            orig_idx = int(m.group(1))
            if orig_idx not in orig_to_new:
                source = next((s for s in sources if s["index"] == orig_idx), None)
                if source:
                    orig_to_new[orig_idx] = {"new_idx": next_new_idx, "source": source}
                    next_new_idx += 1

        # 2. Replace citations and move before punctuation
        def replace_fn(match):
            orig_idx = int(match.group(1))
            if orig_idx in orig_to_new:
                return f"[{orig_to_new[orig_idx]['new_idx']}]"
            return match.group(0)

        formatted_answer = re.sub(cite_pattern, replace_fn, answer)
        formatted_answer = re.sub(r'\.\s*(\[\d+\])', r' \1.', formatted_answer)

        # 3. Build source footer
        if "Không tìm thấy thông tin" in answer or not orig_to_new:
            return formatted_answer

        cited_items = sorted(orig_to_new.values(), key=lambda x: x["new_idx"])
        source_lines = []
        for item in cited_items:
            s = item["source"]
            source_lines.append(f"[{item['new_idx']}] {s['title']} (Ban hành: {s['issue_date']})")

        return f"{formatted_answer}\n\nNguồn tham khảo:\n" + "\n".join(source_lines)

    def _filter_by_confidence(self, docs: List[Document]) -> List[Document]:
        """Filter documents based on confidence threshold."""
        return [d for d in docs if d.metadata.get("confidence_score", 0.0) >= self.confidence_threshold]

    def _build_context(self, docs: List[Document]) -> Tuple[str, List[Dict]]:
        """Build combined context, sorted newest-first, deduplicated by title/date."""
        unique_docs = []
        seen_sources = set()
        
        for doc in docs:
            # Deduplicate by title and issue_date to ensure one file = one source ID
            title = doc.metadata.get("title", "unknown")
            issue_date = str(doc.metadata.get("issue_date", "N/A"))
            source_key = f"{title}_{issue_date}"
            
            if source_key not in seen_sources:
                unique_docs.append(doc)
                seen_sources.add(source_key)

        # Sort newest-first so Source 1 is always the most recent regulation
        def _parse_ts(doc: Document) -> float:
            date_str = str(doc.metadata.get("issue_date", "")).strip()
            if not date_str or date_str == "N/A":
                return 0.0
            from datetime import datetime
            for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y"):
                try:
                    return datetime.strptime(date_str, fmt).timestamp()
                except ValueError:
                    continue
            return 0.0

        unique_docs.sort(key=_parse_ts, reverse=True)

        raw_sources = []
        for i, doc in enumerate(unique_docs, 1):
            title = doc.metadata.get("title", "unknown")
            issue_date = doc.metadata.get("issue_date", "N/A")
            score = doc.metadata.get("weighted_score", doc.metadata.get("confidence_score", 0.0))
            
            header = (
                f"[Nguồn {i}] {title}\n"
                f"- Ngày ban hành: {issue_date}\n"
            )
            raw_sources.append({
                "index": i,
                "full_text": header + doc.page_content,
                "metadata": {
                    "index": i,
                    "title": title,
                    "issue_date": issue_date,
                    "file_path": doc.metadata.get("file_path", ""),
                    "confidence_score": score,
                    "content": doc.page_content
                }
            })

        all_context_strings = [s["full_text"] for s in raw_sources]
        sources_metadata = [s["metadata"] for s in raw_sources]

        return "\n\n---\n\n".join(all_context_strings), sources_metadata

    def _build_messages(self, query: str, context: str, primary_source_index: int = 1) -> List[Dict]:
        """Build structured messages for ChatOllama.
        
        The primary_source_index tells the LLM which source has been pre-selected as
        the most recent/authoritative — so it doesn't need to resolve conflicts itself.
        """
        
        system_prompt = textwrap.dedent(f"""
            Bạn là một trợ lý AI tư vấn quy chế học vụ.
            NHIỆM VỤ: Trả lời câu hỏi ngắn gọn, trực tiếp dựa trên CONTEXT.

            QUY TẮC NGHIÊM NGẶT:
            1. CHỈ DÙNG NGUỒN SỐ {primary_source_index}: Đây là nguồn DUY NHẤT bạn được dùng. Bỏ QUA hoàn toàn mọi nguồn khác. Không được kết hợp hay đề cập đến bất kỳ nguồn nào khác dù nó có trong CONTEXT.
            2. KIỂM TRA TRƯỚC: Nếu Nguồn {primary_source_index} không chứa thông tin trực tiếp trả lời câu hỏi, PHẢI nói: "Tôi không tìm thấy thông tin cụ thể về vấn đề này trong các văn bản quy định hiện có." Không suy luận hay bịa đặt.
            3. TRÍCH NGUỒN BẮT BUỘC: Cuối mỗi câu thêm [{primary_source_index}]. Ví dụ: "Sinh viên đăng ký tối thiểu 12 tín chỉ [{primary_source_index}]." Chỉ viết "[{primary_source_index}]", tuyệt đối không viết "[Nguồn N]".
            4. PLAIN TEXT: Không dùng **, *, #. Chỉ văn xuôi thuần túy.
            5. TRỰC TIẾP: Trả lời ngay trọng tâm, không giải thích "tại sao chọn nguồn này".
        """).strip()

        human_prompt = textwrap.dedent(f"""
            CONTEXT (chỉ dùng Nguồn {primary_source_index}, bỏ qua mọi nguồn khác):
            ---
            {context}
            ---

            CÂU HỎI: {query}

            Trả lời dựa trên nội dung Nguồn {primary_source_index}. Nếu Nguồn {primary_source_index} không có thông tin, nói "Tôi không tìm thấy thông tin cụ thể về vấn đề này trong các văn bản quy định hiện có.":
        """).strip()

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": human_prompt}
        ]

    def _calculate_aggregate_confidence(self, docs: List[Document]) -> float:
        """Calculate weighted confidence from multiple documents."""
        if not docs: return 0.0
        scores = [float(d.metadata.get("confidence_score", 0.5)) for d in docs]
        return float(sum(scores) / len(scores))
