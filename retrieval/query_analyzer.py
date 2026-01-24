import json
import re
import logging
from typing import Dict, List, Optional

from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)


class QueryAnalyzer:
    """Analyzes query intent and context dependency"""
    
    def __init__(self, llm: ChatOllama):
        """Initialize analyzer with LLM
        
        Args:
            llm: Language model for analysis
        """
        self.llm = llm
    
    def extract_metadata(self, query: str) -> Dict[str, str]:
        """Extract metadata filters from query using regex
        
        Args:
            query: User query
            
        Returns:
            Dictionary of extracted metadata filters
        """
        filters = {}
        query_lower = query.lower()
        
        # Extract specific date (ngày/tháng/năm)
        date_match = re.search(r'(\d{1,2})/(\d{1,2})/(\d{4})', query)
        if date_match:
            day, month, year = date_match.groups()
            filters['issue_date'] = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
        else:
            # Extract year for date range
            year_match = re.search(r'\b(20\d{2})\b', query)
            if year_match:
                year = year_match.group(1)
                filters['issue_date_from'] = f"{year}-01-01"
                filters['issue_date_to'] = f"{year}-12-31"
        
        # Extract academic semester (học kỳ)
        semester_match = re.search(r'(học kỳ|HK|hk)(\s*)(\d{1,3})', query_lower)
        if semester_match:
            filters['academic_semester'] = semester_match.group(3)
        
        # Detect target audience keywords
        if any(kw in query_lower for kw in ['sinh viên', 'đại học', 'chính quy', 'đh']):
            filters['target_hint'] = 'sinh_vien_chinh_quy'
        elif any(kw in query_lower for kw in ['cao học', 'thạc sĩ', 'master', 'caohoc']):
            filters['target_hint'] = 'hoc_vien_cao_hoc'
        elif any(kw in query_lower for kw in ['nghiên cứu sinh', 'ncs', 'phd', 'ts']):
            filters['target_hint'] = 'nghien_cuu_sinh'
        
        return filters
    
    def analyze(self, question: str, conversation_history: Optional[List[Dict]] = None) -> Dict:
        """Analyze query intent using LLM with multi-turn support
        
        Args:
            question: User question
            conversation_history: Previous conversation turns
            
        Returns:
            Analysis result with intent, audience, confidence
        """
        if not conversation_history:
            conversation_history = []
        
        # Extract metadata filters from query
        metadata_filters = self.extract_metadata(question)
        
        # Handle empty history case
        if not conversation_history:
            return {
                "is_followup": False,
                "target_audience": "unknown",
                "enhanced_query": question,
                "reasoning": "No conversation history",
                "confidence": 1.0,
                "filters": metadata_filters
            }
        
        # Use LLM for deeper analysis
        analysis_prompt = self._build_analysis_prompt(question, conversation_history)
        
        try:
            response = self.llm.invoke(analysis_prompt)
            content = self._clean_json_response(response.content)
            result = json.loads(content)
            result["confidence"] = self._estimate_confidence(result)
            result["filters"] = metadata_filters
            return result
        except Exception as e:
            logger.error(f"Query analysis failed: {e}", exc_info=True)
            return {
                "is_followup": False,
                "target_audience": "unknown",
                "enhanced_query": question,
                "reasoning": f"Analysis error: {str(e)}",
                "confidence": 0.0,
                "filters": metadata_filters
            }
    
    def _build_analysis_prompt(self, question: str, conversation_history: List[Dict]) -> str:
        """Build LLM prompt for query analysis
        
        Args:
            question: Current question
            conversation_history: Previous conversation turns
            
        Returns:
            Prompt string for LLM
        """
        # Take last 3 turns for context
        max_history = 3
        recent_history = conversation_history[-max_history:] if len(conversation_history) > max_history else conversation_history
        
        history_text = "\n".join([
            f"Q{i+1}: {turn['question']}\nA{i+1}: {turn['answer'][:200]}..."
            for i, turn in enumerate(recent_history)
        ])
        
        return f"""Bạn là chuyên gia phân tích câu hỏi về quy định đại học Việt Nam.

NHIỆM VỤ: Phân tích câu hỏi người dùng để xác định:
1. Có phải follow-up (liên quan lịch sử) hay câu hỏi mới
2. Đối tượng (sinh viên chính quy/cao học/NCS)
3. Mở rộng câu hỏi với context từ lịch sử

HƯỚNG DẪN:
- Follow-up: Câu hỏi liên quan trực tiếp tới Q/A trước, dùng context từ lịch sử
- Câu mới: Câu hỏi độc lập, không liên quan tới những câu trước
- Target audience: Suy luận từ ngữ cảnh (sinh viên, cao học, NCS)
- Enhanced query: Mở rộng với context từ lịch sử nếu follow-up, hoặc giữ nguyên nếu mới
- Confidence: Từ 0-1, cao nếu xác định rõ, thấp nếu mơ hồ

LỊCH SỬ:
{history_text}

CÂU HỎI MỚI: {question}

Trả về JSON format CHỈ CÓ JSON (không comment):
{{
    "is_followup": true/false,
    "target_audience": "sinh_vien_chinh_quy"|"hoc_vien_cao_hoc"|"nghien_cuu_sinh"|"unknown",
    "enhanced_query": "câu hỏi mở rộng hoặc gốc tùy theo follow-up",
    "reasoning": "giải thích tại sao là follow-up/mới, target_audience là gì",
    "confidence": 0.0-1.0
}}

JSON:"""
    
    def _estimate_confidence(self, result: Dict) -> float:
        """Estimate confidence of analysis result based on LLM analysis quality
        
        Args:
            result: Analysis result from LLM
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Start from LLM's own confidence if provided
        confidence = result.get("confidence", 0.5)
        
        # Boost confidence based on analysis quality
        reasoning = result.get("reasoning", "")
        target_audience = result.get("target_audience", "unknown")
        enhanced_query = result.get("enhanced_query", "")
        is_followup = result.get("is_followup", False)
        
        # Penalty for weak reasoning
        if len(reasoning) < 20:
            confidence -= 0.15
        elif len(reasoning) < 50:
            confidence -= 0.05
        
        # Boost for clear target audience
        if target_audience != "unknown":
            confidence += 0.1
        else:
            confidence -= 0.1
        
        # Boost for clear follow-up detection
        if is_followup and len(reasoning) > 30:
            confidence += 0.1
        
        # Boost for meaningful query enhancement
        if enhanced_query and len(enhanced_query) > 20:
            confidence += 0.05
        
        # Penalty for error keywords
        if any(err_word in reasoning.lower() for err_word in ['error', 'fail', 'không', 'lỗi']):
            confidence -= 0.2
        
        return max(0.0, min(1.0, confidence))
    
    def _clean_json_response(self, content: str) -> str:
        """Remove markdown formatting from JSON response
        
        Args:
            content: Raw response content
            
        Returns:
            Cleaned JSON string
        """
        content = content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        return content
    
    def _print_analysis(self, result: Dict) -> None:
        """Display analysis results for debugging
        
        Args:
            result: Analysis result dictionary
        """
        pass