import logging
import re
import textwrap
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from config import Config

logger = logging.getLogger(__name__)

class ResponseGenerator:
    """Modular Response Generator with multi-source grounding and Vietnamese prompts."""

    def __init__(self, config: Dict):
        """Initialize response generator
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.confidence_threshold = Config.CONFIDENCE_THRESHOLD

    def _get_llm(self) -> ChatOllama:
        """Create a fresh ChatOllama instance for the current event loop."""
        return ChatOllama(
            model=self.config["llm_model"],
            temperature=self.config.get("llm_temperature", Config.LLM_TEMPERATURE),
            base_url=self.config.get("ollama_base_url", Config.OLLAMA_BASE_URL),
            timeout=120.0
        )

    async def agenerate(
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
            messages = self._build_messages(
                query=query, 
                context=context, 
                conversation_history=conversation_history,
                primary_source_index=1
            )
            llm = self._get_llm()
            logger.info(f"Generating answer with {len(selected_docs)} sources.")
            
            response = await llm.ainvoke(messages)
            answer = response.content.strip() if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            answer = "Hệ thống gặp sự cố khi tạo câu trả lời."

        # 4. Output Processing & Formatting
        final_response = self._format_response(answer, sources)
        
        if clean_mode:
            return {"answer": answer, "confidence": 1.0, "sources": sources}
            
        return {
            "answer": final_response,
            "confidence": selected_docs[0].metadata.get("weighted_score", selected_docs[0].metadata.get("confidence_score", 0.0)) if selected_docs else 0.0,
            "sources": sources,
        }

    async def astream_generate(
        self,
        query: str,
        documents: List[Document],
        conversation_history: str = "",
    ):
        """Stream response tokens using LLM.astream"""
        selected_docs = self._filter_by_confidence(documents)
        if not selected_docs:
            yield "Không tìm thấy quy định liên quan."
            return

        max_docs: int = getattr(Config, "MAX_RESPONSE_DOCS", 4)
        selected_docs = selected_docs[:max_docs]
        context, sources = self._build_context(selected_docs)
        
        # 3. LLM Call
        full_answer = ""
        buffer = ""
        # Dynamic mapping to track sequential indices as they appear in the stream
        orig_to_display: Dict[int, int] = {}
        next_display_idx = 1
        
        try:
            messages = self._build_messages(
                query=query, 
                context=context, 
                conversation_history=conversation_history,
                primary_source_index=1
            )
            llm = self._get_llm()
            
            async for chunk in llm.astream(messages):
                content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                full_answer += content
                buffer += content
                
                # Process buffer to replace tags mid-stream
                while True:
                    match = re.search(r'\[SOURCE_ID_(\d+)\]', buffer)
                    if not match:
                        break
                    
                    start, end = match.span()
                    orig_idx = int(match.group(1))
                    
                    # Sequential re-indexing: assign a new display index on first appearance
                    if orig_idx not in orig_to_display:
                        # Fallback: check if source actually exists
                        source_exists = any(s["index"] == orig_idx for s in sources)
                        if source_exists or len(sources) == 1:
                            orig_to_display[orig_idx] = next_display_idx
                            next_display_idx += 1
                        else:
                            # Hallucinated ID with no sources? Ignore or map to 1 if one exists
                            if sources:
                                orig_to_display[orig_idx] = 1 # Map to first valid source
                            else:
                                # No sources at all? Strip it
                                yield buffer[:start]
                                buffer = buffer[end:]
                                continue

                    display_idx = orig_to_display.get(orig_idx, 1)
                    
                    yield buffer[:start]
                    yield f"[{display_idx}]"
                    buffer = buffer[end:]
                
                # Yield stabilized content from buffer
                if len(buffer) > 20 and '[' not in buffer[-20:]:
                    yield buffer[:-20]
                    buffer = buffer[-20:]
            
            # Final buffer processing
            if buffer:
                # Handle any remaining tags in final buffer
                def _final_replace(m):
                    oid = int(m.group(1))
                    if oid not in orig_to_display and sources:
                        nonlocal next_display_idx
                        orig_to_display[oid] = next_display_idx
                        next_display_idx += 1
                    return f"[{orig_to_display.get(oid, 1)}]"
                
                buffer = re.sub(r'\[SOURCE_ID_(\d+)\]', _final_replace, buffer)
                yield buffer

        except Exception as e:
            logger.error(f"LLM streaming failed: {e}")
            yield "Hệ thống gặp sự cố khi tạo câu trả lời."
            return

        # Use the accumulated mapping to generate consecutive citations in footer
        footer = self._get_source_footer_from_map(orig_to_display, sources)
        if footer:
            yield "\n\n" + footer

    def _get_source_footer_from_map(self, orig_to_display: Dict[int, int], sources: List[Dict]) -> str:
        """Generate footer based on a dynamic mapping created during streaming."""
        if not orig_to_display:
            # If no citations used but we have sources, maybe just show the first one?
            # Usually better to show nothing if not cited, or all if it's a short answer.
            return ""
        
        # Sort items by display index (1, 2, 3...)
        sorted_mappings = sorted(orig_to_display.items(), key=lambda x: x[1])
        
        source_lines = ["Nguồn tham khảo:"]
        added_any = False
        
        for orig_idx, display_idx in sorted_mappings:
            source = next((s for s in sources if s["index"] == orig_idx), None)
            if not source and len(sources) == 1:
                source = sources[0]
            
            if source:
                source_lines.append(f"[{display_idx}] {source['title']} (Ban hành: {source['issue_date']})")
                added_any = True
                
        return "\n".join(source_lines) if added_any else ""

    async def rewrite_query(self, query: str, conversation_history: str) -> str:
        """Rewrite user query into a standalone question based on history."""
        if not conversation_history:
            return query
            
        system_prompt = textwrap.dedent("""
            Nhiệm vụ: Chuyển câu hỏi của người dùng thành một câu hỏi ĐỘC LẬP (standalone) và ĐẦY ĐỦ Ý NGHĨA dựa trên lịch sử hội thoại.
            
            Quy tắc:
            1. Bạn PHẢI trả về câu hỏi đã được viết lại bằng tiếng Việt.
            2. Câu hỏi mới phải chứa đầy đủ ngữ cảnh để có thể thực hiện tìm kiếm tài liệu mà không cần xem lại lịch sử.
            3. Nếu câu hỏi hiện tại đã đầy đủ hoặc không liên quan đến lịch sử, hãy giữ nguyên nó.
            4. Chỉ trả về văn bản của câu hỏi mới, không giải thích gì thêm.
        """).strip()
        
        human_prompt = f"Lịch sử hội thoại:\n{conversation_history}\n\nCâu hỏi mới: {query}"
        
        try:
            llm = self._get_llm()
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": human_prompt}
            ]
            response = await llm.ainvoke(messages)
            rewritten = response.content.strip() if hasattr(response, 'content') else str(response)
            logger.info(f"Query rewritten: '{query}' -> '{rewritten}'")
            return rewritten
        except Exception as e:
            logger.error(f"Query rewriting failed: {e}")
            return query
            
    def _get_source_footer(self, answer: str, sources: List[Dict]) -> str:
        """Extract logic to build the source footer from _format_response."""
        # Collapse citations and handle both old [N] and new [SOURCE_ID_N] formats
        answer = re.sub(r'\[SOURCE_ID_(\d+)\]', r'[\1]', answer)
        answer = re.sub(r'\[Nguồn\s+(\d+)\]\s*\[\1\]', r'[\1]', answer)
        answer = re.sub(r'\[Nguồn\s+(\d+)\]', r'[\1]', answer)
        
        cite_pattern = r'\[(\d+)\]'
        all_matches = list(re.finditer(cite_pattern, answer))
        
        orig_to_new: Dict[int, Dict] = {}
        next_new_idx: int = 1
        for m in all_matches:
            orig_idx = int(m.group(1))
            if orig_idx not in orig_to_new:
                source = next((s for s in sources if s["index"] == orig_idx), None)
                # Fallback: if only one source exists, map any citation to it
                if not source and len(sources) == 1:
                    source = sources[0]
                
                if source:
                    orig_to_new[orig_idx] = {"new_idx": next_new_idx, "source": source}
                    next_new_idx += 1

        if not orig_to_new: return ""
        
        cited_items = sorted(orig_to_new.values(), key=lambda x: x["new_idx"])
        source_lines = ["Nguồn tham khảo:"]
        for item in cited_items:
            s = item["source"]
            source_lines.append(f"[{item['new_idx']}] {s['title']} (Ban hành: {s['issue_date']})")
        return "\n".join(source_lines)

    def _format_response(self, answer: str, sources: List[Dict]) -> str:
        """Format final answer with sequential source indexing."""
        if not sources:
            return answer

        # 0. Pre-collapse: remove duplicate patterns and convert SOURCE_ID_N to [N]
        #    Handle: [SOURCE_ID_1], [Nguồn 1] [1], [Nguồn 1]
        answer = re.sub(r'\[SOURCE_ID_(\d+)\]', r'[\1]', answer)
        answer = re.sub(r'\[Nguồn\s+(\d+)\]\s*\[\1\]', r'[\1]', answer)
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
                # Robustness: if only one source is provided, assume any [N] refers to it
                if not source and len(sources) == 1:
                    source = sources[0]
                
                if source:
                    orig_to_new[orig_idx] = {"new_idx": next_new_idx, "source": source}
                    next_new_idx += 1

        # 2. Replace citations and move before punctuation
        def replace_fn(match):
            orig_idx = int(match.group(1))
            if orig_idx in orig_to_new:
                return f"[{orig_to_new[orig_idx]['new_idx']}]"
            return "" # Remove invalid/hallucinated citations

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
                f"[SOURCE_ID_{i}] {title}\n"
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

    def _build_messages(
        self, 
        query: str, 
        context: str, 
        conversation_history: str = "",
        primary_source_index: int = 1
    ) -> List[Dict]:
        """Build structured messages for ChatOllama with recency awareness."""
        
        system_prompt = textwrap.dedent(f"""
            Bạn là một trợ lý AI tư vấn quy chế học vụ của trường Đại học Bách Khoa - ĐHQG TP.HCM (HCMUT).
            NHIỆM VỤ: Trả lời câu hỏi ngắn gọn, trực tiếp và chính xác dựa trên CONTEXT được cung cấp.

            LỊCH SỬ HỘI THOẠI (để tham khảo ngữ cảnh nếu cần):
            {conversation_history if conversation_history else "Chưa có lịch sử."}

            QUY TẮC ƯU TIÊN THỜI GIAN:
            - CONTEXT có thể chứa nhiều văn bản (Nguồn 1, Nguồn 2, ...).
            - Nếu các nguồn có thông tin mâu thuẫn hoặc khác nhau về cùng một vấn đề, bạn PHẢI ưu tiên thông tin từ nguồn có **Ngày ban hành** gần đây nhất (mới nhất).
            - Luôn coi văn bản mới hơn là văn bản cập nhật hoặc thay thế cho văn bản cũ.

            QUY TẮC TRẢ LỜI:
            1. TRÍCH NGUỒN BẮT BUỘC: Cuối mỗi câu/ý phải ghi ký hiệu nguồn dưới dạng [SOURCE_ID_N]. Ví dụ: "Sinh viên đăng ký tối thiểu 12 tín chỉ [SOURCE_ID_1]."
            2. CHỈ SỬ DỤNG NHÃN CÓ TRONG CONTEXT: Tuyệt đối không tự bịa ra nhãn hoặc dùng số thứ tự khác ngoài các nhãn [SOURCE_ID_1], [SOURCE_ID_2]... đã được cung cấp.
            3. PLAIN TEXT: Không dùng format markdown như **, *, #. Chỉ dùng văn xuôi thuần túy.
            4. TRUNG THỰC: Nếu không tìm thấy thông tin trong CONTEXT, hãy nói: "Tôi không tìm thấy thông tin cụ thể về vấn đề này trong các văn bản quy định hiện có."
            5. NGẮN GỌN: Đi thẳng vào vấn đề, không giải thích dài dòng về cách bạn chọn nguồn.
            6. TỔNG HỢP: TUYỆT ĐỐI KHÔNG ĐƯỢC TRẢ LỜI TỔNG HỢP TỪ NHIỀU NGUỒN, CHỈ ĐƯỢC TRẢ LỜI DỰA TRÊN 1 NGUỒN DUY NHẤT VÀ ĐÚNG NHẤT.
        """).strip()

        human_prompt = textwrap.dedent(f"""
            CONTEXT:
            ---
            {context}
            ---

            CÂU HỎI: {query}

            Dựa trên CONTEXT (ưu tiên nguồn mới nhất nếu có mâu thuẫn), hãy trả lời câu hỏi trên:
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
