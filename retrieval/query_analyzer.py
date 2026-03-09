import logging
import json
import re
from typing import Dict, List, Optional
from langchain_ollama import ChatOllama
from config import Config

logger = logging.getLogger(__name__)

class QueryAnalyzer:
    """Modular Query Analyzer for intention detection and query enhancement."""

    def __init__(self, llm: ChatOllama):
        self.llm = llm

    def analyze(self, query: str, history: str = "") -> Dict:
        """Analyze user query to determine intent, filters, and enhanced version."""
        
        # Try metadata-based extraction first
        dates = self._extract_dates(query)
        audience = self._detect_audience(query)
        
        prompt = self._build_prompt(query, history)
        
        try:
            response = self.llm.invoke(prompt)
            result = self._parse_json_response(response.content)
            
            # Merge with heuristic extractions if LLM missed them
            if not result.get("issue_dates") and dates:
                result["issue_dates"] = dates
            if result.get("target_audience") == "unknown" and audience:
                result["target_audience"] = audience
                
            return result
        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            return self._get_fallback_analysis(query, audience, dates)

    def _build_prompt(self, query: str, history: str) -> str:
        return f"""Bạn là chuyên gia phân tích yêu cầu cho hệ thống RAG về quy chế học vụ.
Hãy phân tích câu hỏi sau:
Câu hỏi: {query}
Lịch sử hội thoại: {history}

Hướng dẫn:
- "target_audience": sinh_vien_chinh_quy, hoc_vien_cao_hoc, nghien_cuu_sinh.
- "filters": {{ "doc_type": "DTDH" hoặc "DTSDH", "regulation_type": "QDHV" hoặc null }}.
- "enhanced_query": Viết lại câu hỏi đầy đủ, rõ ràng bằng tiếng Việt phổ thông, giải thích các từ viết tắt (SV, TC, HK, HK232 -> Học kỳ 2 năm học 2023-2024, v.v.) để tối ưu tìm kiếm.

Trả về JSON duy nhất với các trường:
1. "intent": Loại ý định (tra_cuu_quy_dinh, giai_thich_quy_trinh, khac).
2. "target_audience": Đối tượng liên quan.
3. "enhanced_query": Câu hỏi đã mở rộng.
4. "filters": Bộ lọc metadata chính xác.
5. "confidence": Độ tin cậy của phân tích (0.0 - 1.0).

Chỉ trả về JSON.
JSON:"""

    def _parse_json_response(self, text: str) -> Dict:
        """Robust JSON parsing for LLM output."""
        try:
            # Try to find JSON block
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group())
            return json.loads(text)
        except Exception:
            # Manual regex fallback if JSON is malformed
            return {
                "intent": "tra_cuu_quy_dinh",
                "target_audience": "unknown",
                "enhanced_query": text[:100],
                "filters": {},
                "confidence": 0.5
            }

    def _extract_dates(self, text: str) -> List[str]:
        """Extract dates from text using predefined patterns."""
        found_dates = []
        for pattern in Config.DATE_PATTERNS:
            matches = re.findall(pattern, text)
            for m in matches:
                # Format to YYYY-MM-DD
                if len(m) == 3:
                    d, m, y = m
                    if len(y) == 2: y = "20" + y
                    found_dates.append(f"{y}-{m.zfill(2)}-{d.zfill(2)}")
        return found_dates

    def _detect_audience(self, text: str) -> Optional[str]:
        """Heuristic-based audience detection."""
        text = text.lower()
        if any(w in text for w in ["sinh viên", "sv", "chính quy"]):
            return "sinh_vien_chinh_quy"
        if any(w in text for w in ["cao học", "thạc sĩ", "học viên"]):
            return "hoc_vien_cao_hoc"
        if any(w in text for w in ["nghiên cứu sinh", "ncs", "tiến sĩ"]):
            return "nghien_cuu_sinh"
        return None

    def _get_fallback_analysis(self, query: str, audience: str, dates: List[str]) -> Dict:
        return {
            "intent": "tra_cuu_quy_dinh",
            "target_audience": audience or "unknown",
            "enhanced_query": query,
            "filters": {},
            "issue_dates": dates,
            "confidence": 0.3
        }
