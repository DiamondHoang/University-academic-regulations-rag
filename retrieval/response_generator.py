import re
import textwrap
from datetime import datetime
from typing import List, Dict, Tuple, Any, Optional, AsyncGenerator

from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from config import Config

class ResponseGenerator:
    """Generates grounded responses based on university academic regulations."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize generator with system-wide configuration."""
        self.config = config
        self.confidence_threshold = Config.CONFIDENCE_THRESHOLD
        self._llm: Optional[Any] = None  # Cached LLM client

    def _get_llm(self) -> Any:
        """Get or initialize the LLM client (Factory Pattern)."""
        if self._llm:
            return self._llm

        provider = str(self.config.get("llm_provider", Config.LLM_PROVIDER)).lower()
        temperature = self.config.get("llm_temperature", Config.LLM_TEMPERATURE)
        
        print(f"[LLM] Initializing {provider}")

        if provider == "groq":
            self._llm = ChatGroq(
                model=self.config.get("groq_model", Config.GROQ_MODEL),
                api_key=self.config.get("groq_api_key", Config.GROQ_API_KEY),
                temperature=temperature,
                max_retries=3,  # Native retry handling
            )
        else:
            self._llm = ChatOllama(
                base_url=self.config.get("ollama_base_url", Config.OLLAMA_BASE_URL),
                model=self.config.get("ollama_model", Config.OLLAMA_MODEL),
                temperature=temperature,
                timeout=30.0
            )
        return self._llm

    @staticmethod
    def _classify_llm_error(error_msg: str, provider: str = "LLM") -> str:
        """Map technical exceptions to professional Vietnamese messages."""
        print(f"[ERROR] [LLM] ({provider}): {error_msg}")
        
        err_lower = error_msg.lower()
        if "401" in err_lower or "api_key" in err_lower:
            return f"Lỗi xác thực: API Key {provider} không chính xác hoặc đã hết hạn."
        if "429" in err_lower or "quota" in err_lower or "limit" in err_lower:
            return "Hệ thống đang tạm thời quá tải (Rate Limit). Vui lòng thử lại sau giây lát."
        if "connection" in err_lower or "refused" in err_lower:
            return f"Lỗi kết nối: Không thể kết nối tới máy chủ {provider}."
        if "timeout" in err_lower:
            return "Hệ thống phản hồi quá chậm (Timeout). Vui lòng thử lại sau."
            
        return "Xin lỗi, tôi gặp sự cố kỹ thuật khi xử lý câu trả lời. Vui lòng thử lại sau."

    async def agenerate(
        self,
        query: str,
        documents: List[Document],
        conversation_history: str = "",
        history_messages: Optional[List[Dict[str, str]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate a complete grounded answer."""
        selected_docs = [d for d in documents if d.metadata.get("confidence_score", 0.0) >= self.confidence_threshold]
        
        if not selected_docs:
            return {
                "answer": f"Xin lỗi, tôi không tìm thấy thông tin chính thức về '{query}' trong quy định hiện tại.",
                "confidence": 0.0,
                "sources": []
            }

        max_docs: int = getattr(Config, "MAX_RESPONSE_DOCS", 4)
        selected_docs = selected_docs[:max_docs]
        context, sources = self._build_context(selected_docs)

        try:
            messages = self._build_messages(
                query=query, 
                context=context, 
                conversation_history=conversation_history,
                history_messages=history_messages,
            )
            llm = self._get_llm()
            
            response = await llm.ainvoke(messages)
            answer = response.content.strip() if hasattr(response, 'content') else str(response)
        except Exception as e:
            provider = str(self.config.get("llm_provider", Config.LLM_PROVIDER) or "llm")
            return {
                "answer": self._classify_llm_error(str(e), provider),
                "confidence": 0.0,
                "sources": sources
            }

        formatted_answer = self._format_response(answer, sources)
        score = selected_docs[0].metadata.get("confidence_score", 0.0)
        return {"answer": formatted_answer, "confidence": score, "sources": sources}

    async def astream_generate(
        self,
        query: str,
        documents: List[Document],
        conversation_history: str = "",
        history_messages: Optional[List[Dict[str, str]]] = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Stream generated response with real-time citation replacement."""
        selected_docs = [d for d in documents if d.metadata.get("confidence_score", 0.0) >= self.confidence_threshold]
        if not selected_docs:
            yield "Không tìm thấy quy trình/quy định phù hợp trong hệ thống văn bản."
            return

        max_docs: int = getattr(Config, "MAX_RESPONSE_DOCS", 4)
        selected_docs = selected_docs[:max_docs]
        context, sources = self._build_context(selected_docs)
        
        full_answer = ""
        buffer = ""
        citation_map: Dict[int, int] = {}
        next_display_idx = 1
        
        try:
            messages = self._build_messages(
                query=query, 
                context=context, 
                conversation_history=conversation_history,
                history_messages=history_messages,
            )
            llm = self._get_llm()
            
            async for chunk in llm.astream(messages):
                content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                
                # Detect the start of a manual source list from LLM to truncate it
                if re.search(r'\n*(Nguồn tham khảo|References)\s*:', content, re.IGNORECASE) or \
                   re.search(r'\n*(Nguồn tham khảo|References)\s*:', full_answer + content, re.IGNORECASE):
                    full_answer += content
                    break

                full_answer += content
                buffer += content
                
                # Replace markers [SOURCE_ID_N] -> [N] in stream
                while True:
                    match = re.search(r'\[SOURCE_ID_(\d+)\]', buffer)
                    if not match:
                        break
                    
                    start, end = match.span()
                    orig_idx = int(match.group(1))
                    
                    if orig_idx not in citation_map:
                        # Ensure ID actually corresponds to a source
                        if any(s["index"] == orig_idx for s in sources):
                            citation_map[orig_idx] = next_display_idx
                            next_display_idx += 1
                        else:
                            # Hallucinated ID mapping fallback
                            citation_map[orig_idx] = 1 if sources else 0

                    display_idx = citation_map.get(orig_idx, 1)
                    yield buffer[:start]
                    yield f"[{display_idx}]"
                    buffer = buffer[end:]
                
                # Buffer management for partial markers
                if len(buffer) > 20 and '[' not in buffer[-20:]:
                    yield buffer[:-20]
                    buffer = buffer[-20:]
            
            if buffer:
                # Handle remnants
                def _final_rep(m):
                    oid = int(m.group(1))
                    return f"[{citation_map.get(oid, 1)}]"
                buffer = re.sub(r'\[SOURCE_ID_(\d+)\]', _final_rep, buffer)
                yield buffer

        except Exception as e:
            provider = str(self.config.get("llm_provider", Config.LLM_PROVIDER) or "llm")
            yield self._classify_llm_error(str(e), provider)
            return

        # Append source footer
        footer = self._get_source_footer_from_map(citation_map, sources)
        if footer:
            yield "\n\n" + footer

    def _get_source_footer_from_map(self, citation_map: Dict[int, int], sources: List[Dict[str, Any]]) -> str:
        """Create a footer list using only successfully cited sources."""
        if not citation_map: return ""
        
        sorted_cites = sorted(citation_map.items(), key=lambda x: x[1])
        output = ["Nguồn tham khảo:"]
        
        for orig_idx, display_idx in sorted_cites:
            src = next((s for s in sources if s["index"] == orig_idx), None)
            if src:
                output.append(f"[{display_idx}] {src['title']} (Ban hành: {src['issue_date']})")
                
        return "\n".join(output) if len(output) > 1 else ""

    async def rewrite_query(self, query: str, conversation_history: str, history_messages: Optional[List[Dict[str, Any]]] = None, **kwargs) -> str:
        """Restructure query to be standalone based on conversation context."""
        if not conversation_history and not history_messages:
            return query
            
        system = textwrap.dedent("""
            Nhiệm vụ: Viết lại câu hỏi của người dùng thành một câu hỏi ĐỘC LẬP và đầy đủ ý nghĩa dựa vào lịch sử hội thoại.
            Yêu cầu:
            - Giữ nguyên ngôn ngữ tiếng Việt.
            - Chỉ trả về câu hỏi đã viết lại, không giải thích.
            - Nếu câu hỏi đã đầy đủ ý nghĩa, giữ nguyên nội dung.
        """).strip()
        
        history_str = "\n".join([f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}" for m in (history_messages or [])]) or conversation_history
        human = f"Lịch sử:\n{history_str}\n\nCâu hỏi mới: {query}"
        
        try:
            llm = self._get_llm()
            print(f"[LLM] Rewriting query context")
            resp = await llm.ainvoke([{"role": "system", "content": system}, {"role": "user", "content": human}])
            return resp.content.strip() if hasattr(resp, 'content') else str(resp)
        except Exception:
            # Simple fallback: combine current with potentially relevant previous topic
            return query

    def _format_response(self, answer: str, sources: List[Dict[str, Any]]) -> str:
        """Normalize markers and append footer for non-streaming response."""
        answer = re.sub(r'\[SOURCE_ID_(\d+)\]', r'[\1]', answer)
        
        citation_map = {}
        next_idx = 1
        
        def _remap(match):
            nonlocal next_idx
            orig_id = int(match.group(1))
            if orig_id not in citation_map:
                if any(s["index"] == orig_id for s in sources):
                    citation_map[orig_id] = next_idx
                    next_idx += 1
                else: return ""
            return f"[{citation_map[orig_id]}]"

        answer = re.sub(r'\[(\d+)\]', _remap, answer)
        footer = self._get_source_footer_from_map(citation_map, sources)
        return (answer + "\n\n" + footer).strip()

    def _build_context(self, docs: List[Document]) -> Tuple[str, List[Dict[str, Any]]]:
        """Aggregate unique documents and annotate with SOURCE_ID markers."""
        unique_docs = []
        seen = set()
        for doc in docs:
            key = f"{doc.metadata.get('title')}_{doc.metadata.get('issue_date')}"
            if key not in seen:
                unique_docs.append(doc)
                seen.add(key)

        # Sort newest-first (Source 1 is most recent)
        unique_docs.sort(key=lambda d: str(d.metadata.get("issue_date", "1970-01-01")), reverse=True)

        context_blocks = []
        sources_meta = []
        for i, doc in enumerate(unique_docs, 1):
            title = doc.metadata.get("title", "Quy định")
            date = doc.metadata.get("issue_date", "N/A")
            
            context_blocks.append(f"[SOURCE_ID_{i}] {title} (Ban hành: {date})\nInterpreted Content: {doc.page_content}")
            sources_meta.append({
                "index": i,
                "title": title,
                "issue_date": date,
            })

        return "\n\n---\n\n".join(context_blocks), sources_meta

    def _build_messages(self, query: str, context: str, conversation_history: str = "", history_messages: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, str]]:
        """Construct prompt messages with strict system instructions."""
        system = textwrap.dedent("""
            Bạn là trợ lý tư vấn quy chế học vụ trường ĐH Bách Khoa HCM.
            PHẢI tuân thủ các quy tắc sau:
            1. Trả lời CHỈ DỰA TRÊN CONTEXT được cung cấp.
            2. ƯU TIÊN NGUỒN MỚI: [SOURCE_ID_1] là văn bản mới nhất. Nếu [SOURCE_ID_1] có thông tin, hãy dùng nó và bỏ qua các văn bản cũ hơn.
            3. TRÍCH DẪN: Phải ghi [SOURCE_ID_N] ở cuối câu chứa thông tin đó.
            4. THÀNH THẬT: Nếu không có thông tin trong Context, nói rõ "Tôi không tìm thấy thông tin cụ thể".
            5. GỌN GÀNG: Chỉ trả về nội dung quy chế liên quan TRỰC TIẾP đến câu hỏi hiện tại.
            6. KHÔNG LẶP LẠI: Tuyệt đối không nhắc lại các thông tin đã trả lời ở phần Lịch sử nếu câu hỏi hiện tại không yêu cầu tổng hợp lại.
        """).strip()

        history_str = conversation_history if not history_messages else ""
        user_content = f"CONTEXT:\n---\n{context}\n---\n\nLỊCH SỬ:\n{history_str}\n\nCÂU HỎI: {query}\n\nHãy trả lời chuyên nghiệp và trung thực:"
        
        msgs = [{"role": "system", "content": system}]
        if history_messages: msgs.extend(history_messages)
        msgs.append({"role": "user", "content": user_content})
        return msgs
