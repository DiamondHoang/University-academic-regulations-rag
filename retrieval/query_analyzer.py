import json
import re
import logging
from typing import Dict, List, Optional

from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)


class QueryAnalyzer:
    """Analyze user queries for intent, follow-up, and metadata filtering"""

    def __init__(self, llm: ChatOllama):
        self.llm = llm

    # ======================
    # METADATA EXTRACTION
    # ======================
    def extract_metadata(self, query: str) -> Dict[str, str]:
        """Extract structured filters using lightweight rules"""
        filters = {}
        query_lower = query.lower()

        # ---- Issue date ----
        date_match = re.search(r'(\d{1,2})/(\d{1,2})/(\d{4})', query)
        if date_match:
            day, month, year = date_match.groups()
            filters["issue_date"] = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
        else:
            year_match = re.search(r'\b(20\d{2})\b', query)
            if year_match:
                year = year_match.group(1)
                filters["issue_date_from"] = f"{year}-01-01"
                filters["issue_date_to"] = f"{year}-12-31"

        # ---- Target audience hint ----
        filters["target_hint"] = self._detect_audience(query_lower)

        return filters

    def _detect_audience(self, query_lower: str) -> str:
        """Rule-based audience detection"""
        if any(kw in query_lower for kw in ["sinh viên", "đại học", "chính quy", "đh"]):
            return "sinh_vien_chinh_quy"
        if any(kw in query_lower for kw in ["cao học", "thạc sĩ", "master"]):
            return "hoc_vien_cao_hoc"
        if any(kw in query_lower for kw in ["nghiên cứu sinh", "ncs", "phd"]):
            return "nghien_cuu_sinh"
        return "unknown"

    # ======================
    # MAIN ANALYSIS
    # ======================
    def analyze(
        self,
        question: str,
        conversation_history: Optional[List[Dict]] = None
    ) -> Dict:

        if not conversation_history:
            conversation_history = []

        metadata_filters = self.extract_metadata(question)

        # ---- Fast path: no history ----
        if not conversation_history:
            return {
                "is_followup": False,
                "target_audience": metadata_filters.get("target_hint", "unknown"),
                "enhanced_query": question,
                "reasoning": "No history",
                "confidence": 1.0,
                "filters": metadata_filters,
            }

        # ---- LLM reasoning ----
        prompt = self._build_analysis_prompt(question, conversation_history)

        try:
            response = self.llm.invoke(prompt)
            raw = getattr(response, "content", "") or ""
            content = self._clean_json_response(raw)

            if not content:
                raise ValueError("Empty response content from LLM")

            # attempt to parse JSON; be forgiving if the model returns junk
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                try:
                    extracted = self._extract_json_from_text(content)
                    result = json.loads(extracted)
                except Exception as inner_e:
                    # log once and fall back to a safe default object
                    logger.warning(
                        "Failed to parse JSON from analysis response, using fallback.\n"
                        "raw: %s\ncleaned: %s\nerror: %s",
                        raw[:200], content[:200], inner_e
                    )
                    result = {
                        "is_followup": False,
                        "target_audience": "unknown",
                        "enhanced_query": question,
                        "reasoning": "model returned invalid JSON",
                        "confidence": 0.0,
                    }
            # deterministic confidence calibration (only if not already set)
            if "confidence" not in result:
                result["confidence"] = self._estimate_confidence(result)

        except Exception as e:
            logger.error(
                "Query analysis failed: %s; response preview: %s",
                e,
                (raw[:500] + "...") if 'raw' in locals() and raw else "<no response>",
                exc_info=True,
            )
            result = {
                "is_followup": False,
                "target_audience": "unknown",
                "enhanced_query": question,
                "reasoning": str(e),
                "confidence": 0.0,
            }

        result["filters"] = metadata_filters
        return result

    # ======================
    # PROMPT
    # ======================
    def _build_analysis_prompt(
        self,
        question: str,
        conversation_history: List[Dict],
    ) -> str:

        history = conversation_history[-3:]

        history_text = "\n".join(
            f"Q{i+1}: {turn['question']}\nA{i+1}: {turn['answer'][:150]}..."
            for i, turn in enumerate(history)
        )

        return f"""Bạn là chuyên gia phân tích câu hỏi về quy định đại học Việt Nam.

Hướng dẫn cho LLM:
- LUÔN TRẢ VỀ DUY NHẤT 1 ĐỐI TƯỢNG JSON hợp lệ, KHÔNG thêm lời giải thích.
- KHÔNG giải toán, KHÔNG tạo ví dụ, chỉ phân tích văn bản câu hỏi.
- Nếu không thể phân tích, trả về JSON mặc định với giá trị phù hợp.

Phân tích cần thực hiện:
1. Là câu hỏi tiếp nối hay mới
2. Đối tượng (sinh viên/chuyên ngành ...)
3. Viết lại câu hỏi nếu cần (enhanced_query)

LỊCH SỬ:
{history_text}

CÂU HỎI: {question}

Trả về JSON (như mẫu):
{{
  "is_followup": true/false,
  "target_audience": "...",
  "enhanced_query": "...",
  "reasoning": "...",
  "confidence": 0.0-1.0
}}
"""

    # ======================
    # CONFIDENCE
    # ======================
    def _estimate_confidence(self, result: Dict) -> float:
        """Calibrate LLM confidence using heuristic signals"""

        base = result.get("confidence", 0.5)

        # Strong reasoning signal
        if len(result.get("reasoning", "")) > 40:
            base += 0.1

        # Target clarity
        if result.get("target_audience") != "unknown":
            base += 0.1
        else:
            base -= 0.1

        # Query enrichment
        if result.get("enhanced_query") != "":
            base += 0.05

        return max(0.0, min(1.0, base))

    # ======================
    # UTILS
    # ======================
    def _clean_json_response(self, content: str) -> str:
        content = content.strip()
        if "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        return content

    def _extract_json_from_text(self, text: str) -> str:
        """Attempt to extract a JSON object or array substring from arbitrary text.

        Looks for the first/fullest {...} or [...] span and returns it. Raises
        ValueError if no plausible JSON span is found.
        """
        # Try object
        obj_start = text.find("{")
        obj_end = text.rfind("}")
        if obj_start != -1 and obj_end != -1 and obj_end > obj_start:
            return text[obj_start:obj_end + 1]

        # Try array
        arr_start = text.find("[")
        arr_end = text.rfind("]")
        if arr_start != -1 and arr_end != -1 and arr_end > arr_start:
            return text[arr_start:arr_end + 1]

        raise ValueError("No JSON object or array found in LLM response")
