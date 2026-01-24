import re
from typing import List, Dict, Optional, Tuple
import logging

from langchain_core.documents import Document
from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)


class ResponseGenerator:
    """Generate answers with proper citations from documents"""
    
    MAX_DOCS_IN_CONTEXT = 5
    MAX_CONTENT_LENGTH = 1500
    MIN_ANSWER_LENGTH = 100
    
    def __init__(self, llm: ChatOllama, max_context_length: int = 8000):
        """Initialize response generator
        
        Args:
            llm: Language model for answer generation
            max_context_length: Maximum length of context to use
        """
        self.llm = llm
        self.max_context_length = max_context_length
    
    def generate(
        self,
        question: str,
        documents: List[Document],
        conversation_history: str = "",
        analysis_result: Optional[Dict] = None,
        clean_mode: bool = False
    ) -> Dict:
        """Generate answer with citations from documents
        
        Args:
            question: User question
            documents: Retrieved documents
            conversation_history: Previous conversation context
            analysis_result: Query analysis result
            clean_mode: If True, return clean answer without citations for dataset
            
        Returns:
            Dictionary with answer, confidence, and sources
        """
        try:
            if not documents:
                return {
                    "answer": "Xin lỗi, tôi không tìm thấy thông tin liên quan.",
                    "confidence": 0.0,
                    "sources": []
                }
            
            # Use clean mode for dataset answers
            if clean_mode:
                return self._generate_clean(question, documents)
            
            context, source_info = self._build_context(documents)
            prompt = self._build_prompt(question, context, conversation_history, analysis_result)
            
            response = self.llm.invoke(prompt)
            answer = response.content
            
            # Extract cited sources from answer
            cited_sources = self._extract_cited_sources(answer, source_info)
            
            # Create mapping from old labels to new sequential labels
            label_mapping = {}
            for idx, src in enumerate(cited_sources, 1):
                old_label = src.get('label', '')
                new_label = f"[Nguồn {idx}]"
                label_mapping[old_label] = new_label
                src['label'] = new_label
            
            # Replace old labels with new sequential labels in answer
            for old_label, new_label in label_mapping.items():
                answer = answer.replace(old_label, new_label)
            
            # Remove content from non-cited sources
            cited_labels = {src.get('label', '') for src in cited_sources}
            for src in source_info:
                if src.get('label', '') not in cited_labels:
                    src.pop('content', None)
            
            # Format only cited sources
            sources_text = self._format_sources(cited_sources)
            full_answer = answer + sources_text
            
            # Calculate quality score and extract citations
            quality_score = self._quality_check(answer, source_info)
            
            return {
                "answer": full_answer,
                "confidence": quality_score,
                "sources": source_info
            }
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}", exc_info=True)
            return {
                "answer": f"Xin lỗi, có lỗi xảy ra: {str(e)}",
                "confidence": 0.0,
                "sources": []
            }
    
    def _generate_clean(
        self,
        question: str,
        documents: List[Document]
    ) -> Dict:
        """Generate clean answer without citations for dataset
        
        Uses the same powerful _build_prompt logic but cleans output
        to remove citations, sources, and metadata.
        
        Args:
            question: User question
            documents: Retrieved documents
            
        Returns:
            Dictionary with clean answer
        """
        try:
            context, source_info = self._build_context(documents)
            prompt = self._build_prompt(question, context, "", None)
            
            response = self.llm.invoke(prompt)
            answer = response.content.strip()
            
            # Aggressive cleaning for dataset
            answer = self._clean_answer_for_dataset(answer)
            
            return {
                "answer": answer,
                "confidence": 0.8 if answer else 0.0,
                "sources": []
            }
            
        except Exception as e:
            logger.error(f"Clean response generation failed: {e}", exc_info=True)
            return {
                "answer": f"Xin lỗi, có lỗi xảy ra: {str(e)}",
                "confidence": 0.0,
                "sources": []
            }
    
    def _clean_answer_for_dataset(self, answer: str) -> str:
        """Clean answer by removing all citations, sources, and metadata
        
        Args:
            answer: Raw answer with citations and sources
            
        Returns:
            Clean answer without citations or metadata
        """
        # Remove [Nguồn X], [Nguồn X - text], and other citation patterns
        answer = re.sub(r'\s*\[Nguồn\s+\d+[^\]]*\]', '', answer)
        
        # Remove NGUỒN THAM KHẢO section and everything after
        answer = re.sub(r'\s*NGUỒN THAM KHẢO.*$', '', answer, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove "Cập nhật từ..." phrases and year info (case insensitive)
        answer = re.sub(r'\s*cập nhật từ\s+[^.]*\.', '', answer, flags=re.IGNORECASE)
        answer = re.sub(r'\s*và năm không xác định\.?', '', answer, flags=re.IGNORECASE)
        
        # Remove any dates like "năm 2022", "năm 2023", etc
        answer = re.sub(r'\s*năm\s+\d{4}', '', answer, flags=re.IGNORECASE)
        
        # Remove academic year patterns like "học kỳ 222", "HK231", etc
        answer = re.sub(r'\s*học kỳ\s+\d+', '', answer, flags=re.IGNORECASE)
        answer = re.sub(r'\s*HK\d+', '', answer, flags=re.IGNORECASE)
        
        # Remove common metadata patterns
        answer = re.sub(r'\s*theo\s+\w+\s+\d+', '', answer, flags=re.IGNORECASE)
        
        # Remove trailing commas and periods with metadata
        answer = re.sub(r'\s*,\s*năm\s+\d{4}.*$', '', answer, flags=re.DOTALL)
        
        # Remove newlines and normalize spaces
        answer = re.sub(r'\s*\n\s*', ' ', answer)
        answer = re.sub(r'\s+', ' ', answer)
        
        # Strip leading/trailing whitespace
        return answer.strip()
    
    def _build_context(self, documents: List[Document]) -> Tuple[str, List[Dict]]:
        """Build context string with source tracking and deduplication by title
        
        Args:
            documents: Documents to include in context
            
        Returns:
            Tuple of (context_string, source_info_list)
        """
        # Sort by issue_date (newest first)
        sorted_docs = sorted(
            documents,
            key=lambda d: d.metadata.get('issue_date', '1900-01-01'),
            reverse=True
        )
        
        conflict_groups = self._detect_conflicts(sorted_docs)
        
        # Group documents by title AND doc_type to avoid mixing DTDH and DTSDH
        title_groups = {}
        for doc in sorted_docs[:self.MAX_DOCS_IN_CONTEXT * 2]:  # Get more to compensate for grouping
            title = doc.metadata.get('title', 'N/A')
            doc_type = doc.metadata.get('doc_type', 'unknown')
            group_key = f"{title}||{doc_type}"  # Composite key
            
            if group_key not in title_groups:
                title_groups[group_key] = []
            title_groups[group_key].append(doc)
        
        # Build context and source_info
        context_parts = []
        source_info = []
        source_idx = 1
        
        # Limit to MAX_DOCS_IN_CONTEXT unique sources
        for group_key, docs in list(title_groups.items())[:self.MAX_DOCS_IN_CONTEXT]:
            # Get the newest document in this group (already sorted)
            primary_doc = docs[0]
            title = primary_doc.metadata.get('title', 'N/A')
            doc_type = primary_doc.metadata.get('doc_type', 'unknown')
            doc_type_vn = "Đại học" if doc_type == 'DTDH' else "Sau đại học"
            source_label = f"[Nguồn {source_idx}]"
            
            # Check for conflicts
            conflict_note = ""
            for group in conflict_groups:
                if primary_doc in group:
                    conflict_note = " (Xung đột với nguồn khác)"
                    break
            
            # Check for multiple versions (same title, different dates)
            dates = {doc.metadata.get('issue_date', 'N/A') for doc in docs}
            if len(dates) > 1:
                logger.warning(f"Multiple versions found for '{title}': {dates}")
                conflict_note += " (Nhiều phiên bản)"
            
            # Combine content from all docs in this group
            combined_content_parts = []
            for doc in docs[:3]:  # Limit to 3 chunks per source
                content = doc.page_content
                if len(content) > self.MAX_CONTENT_LENGTH:
                    content = content[:self.MAX_CONTENT_LENGTH] + "..."
                combined_content_parts.append(content)
            
            combined_content = "\n---\n".join(combined_content_parts)
            
            # Add to source_info
            source_info.append({
                "label": source_label,
                "title": title,
                "doc_type": doc_type_vn,
                "regulation_type": primary_doc.metadata.get('regulation_type', 'N/A'),
                "academic_year": primary_doc.metadata.get('academic_year', 'N/A'),
                "issue_date": primary_doc.metadata.get('issue_date', 'N/A'),
                "has_conflict": bool(conflict_note),
                "content": combined_content,
                "num_chunks": len(docs)
            })
            
            # Build context text
            year = primary_doc.metadata.get('academic_year', 'N/A')
            meta_info = f"{source_label} - {title} ({doc_type_vn} - {year}){conflict_note}"
            
            context_parts.append(f"{meta_info}\n{combined_content}")
            source_idx += 1
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Truncate if too long
        if len(context) > self.max_context_length:
            context = context[:self.max_context_length] + "\n\n[Context truncated]"
            logger.warning("Context truncated due to length")
        
        return context, source_info
    
    def _detect_conflicts(self, documents: List[Document]) -> List[List[Document]]:
        """Detect potential conflicts between documents
        
        Args:
            documents: Documents to check
            
        Returns:
            List of conflict groups
        """
        groups = {}
        for doc in documents:
            reg_type = doc.metadata.get('regulation_type', 'unknown')
            if reg_type not in groups:
                groups[reg_type] = []
            groups[reg_type].append(doc)
        
        # Return groups with multiple documents from different dates
        conflict_groups = []
        for group in groups.values():
            if len(group) > 1:
                dates = {doc.metadata.get('issue_date', 'N/A') for doc in group}
                if len(dates) > 1:  # Only conflicts if different dates
                    conflict_groups.append(group)
        
        return conflict_groups
    
    def _quality_check(self, answer: str, source_info: List[Dict]) -> float:
        """Check answer quality based on citations and content
        
        Args:
            answer: Generated answer
            source_info: Source information
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        score = 0.5
        
        # Check for explicit citations
        cited_sources = {src['label'] for src in source_info if src['label'] in answer}
        if source_info:
            citation_ratio = len(cited_sources) / len(source_info)
            score += citation_ratio * 0.3
        
        # Penalty for no citations
        if not cited_sources:
            score -= 0.2
        
        # Bonus for good answer length
        if len(answer) > self.MIN_ANSWER_LENGTH:
            score += 0.1
        
        # Penalty for too short answers (possible hallucination)
        if len(answer) < 30:
            score -= 0.2
        
        # Bonus if answer starts with direct answer
        if any(start in answer[:30].lower() for start in ['có', 'không', 'được', 'không được', 'phải', 'không phải']):
            score += 0.05
        
        # Check for uncertainty indicators (good - means LLM is honest)
        uncertainty = sum(1 for word in ['không rõ', 'không có', 'không quy định', 'chưa có'] if word in answer.lower())
        if uncertainty > 0:
            score += 0.05
        
        return max(0.0, min(1.0, score))
    
    def _build_prompt(
        self,
        question: str,
        context: str,
        history: str,
        analysis_result: Optional[Dict] = None
    ) -> str:
        """Build LLM prompt
        
        Args:
            question: User question
            context: Document context
            history: Conversation history
            analysis_result: Query analysis
            
        Returns:
            Prompt string
        """
        history_section = f"LỊCH SỬ:\n{history}\n" if history else ""
        
        # Customize based on target audience
        audience_instruction = ""
        if analysis_result:
            target_audience = analysis_result.get('target_audience', 'unknown')
            if target_audience == 'sinh_vien_chinh_quy':
                audience_instruction = "Ưu tiên DTDH."
            elif target_audience in ['hoc_vien_cao_hoc', 'nghien_cuu_sinh']:
                audience_instruction = "Ưu tiên DTSDH."
        
        confidence_note = ""
        if analysis_result and analysis_result.get('confidence', 1.0) < 0.7:
            confidence_note = "Lưu ý: Câu hỏi có độ tin cậy thấp."
        
        return f"""Bạn là trợ lý pháp lý chuyên trả lời quy định đại học.

VAI TRÒ:
- Trích xuất thông tin CHÍNH XÁC từ văn bản quy định

NGUYÊN TẮC XỬ LÝ NGUỒN:
1. XÁC ĐỊNH HIỆU LỰC:
   - Văn bản có issue_date MỚI HƠN sẽ GHI ĐÈ văn bản cũ về cùng vấn đề
   - Nếu văn bản mới quy định khác/bãi bỏ → CHỈ dùng văn bản mới
   - Nếu văn bản mới KHÔNG ĐỀ CẬP → mới kết hợp cả 2 nguồn

2. PHÁT HIỆN MÂU THUẪN:
   - Nếu 2 nguồn có quy định TRÁI NGƯỢC về cùng vấn đề
   - Ví dụ: Nguồn A "được phép X", Nguồn B "không được phép X"
   → CHỈ trích dẫn nguồn MỚI NHẤT, ghi chú rõ: 
   "Quy định này đã thay đổi so với [Nguồn cũ - năm X]"

3. KẾT HỢP NGUỒN (chỉ khi):
   - Các nguồn bổ sung cho nhau (không mâu thuẫn)
   - Văn bản mới không hủy bỏ quy định cũ
   - Áp dụng cho đối tượng/điều kiện KHÁC NHAU

NGUYÊN TẮC TRÍCH XUẤT:
1. CHỈ sử dụng thông tin có trong văn bản
2. ĐƯỢC PHÉP diễn giải để dễ hiểu, nhưng:
   - Giữ nguyên số liệu, mốc thời gian, điều kiện
   - Chỉ thay đổi cách diễn đạt
   - Không thay đổi phạm vi áp dụng
3. KHÔNG thêm điều kiện/hệ quả/kết luận mới
4. KHÔNG áp dụng logic suy luận ngoài văn bản
5. Nếu không có thông tin → nói rõ "không có quy định về vấn đề này"

HÌNH THỨC TRẢ LỜI:
- Văn bản thuần (không dùng Markdown) dưới dạng đoạn văn
- Mỗi ý có [Nguồn X]
- Nếu có thay đổi quy định, ghi chú: "Cập nhật từ [năm X]"

PHÂN BIỆT ĐỐI TƯỢNG:
- DTDH = đào tạo đại học (sinh viên chính quy)
- DTSDH = đào tạo sau đại học (cao học/NCS)
{audience_instruction}
{confidence_note}

{history_section}
NGỮ CẢNH (CÁC NGUỒN):
{context}

CÂU HỎI:
{question}

YÊU CẦU TRẢ LỜI:
- Trả lời trực tiếp câu hỏi
- Ngắn gọn, chính xác
- Mỗi câu có trích dẫn [Nguồn X] và trích dẫn trước dấu chấm

TRẢ LỜI:"""
    
    def _build_clean_prompt(
        self,
        question: str,
        context: str
    ) -> str:
        """Build prompt for clean, factual answers without citations
        
        Used for dataset questions that need direct, concise answers
        without source citations or metadata.
        Applies same principles as _build_prompt but without citations.
        
        Args:
            question: User question
            context: Document context
            
        Returns:
            Prompt string
        """
        return f"""Bạn là trợ lý pháp lý chuyên trả lời quy định đại học.

VAI TRÒ:
- Trích xuất thông tin CHÍNH XÁC từ văn bản quy định

NGUYÊN TẮC XỬ LÝ NGUỒN:
1. XÁC ĐỊNH HIỆU LỰC:
   - Văn bản có issue_date MỚI HƠN sẽ GHI ĐÈ văn bản cũ về cùng vấn đề
   - Nếu văn bản mới quy định khác/bãi bỏ → CHỈ dùng văn bản mới
   - Nếu văn bản mới KHÔNG ĐỀ CẬP → mới kết hợp cả 2 nguồn

2. PHÁT HIỆN MÂU THUẪN:
   - Nếu 2 nguồn có quy định TRÁI NGƯỢC về cùng vấn đề
   - Ví dụ: Nguồn A "được phép X", Nguồn B "không được phép X"
   → CHỈ dùng quy định MỚI NHẤT

3. KẾT HỢP NGUỒN (chỉ khi):
   - Các nguồn bổ sung cho nhau (không mâu thuẫn)
   - Văn bản mới không hủy bỏ quy định cũ
   - Áp dụng cho đối tượng/điều kiện KHÁC NHAU

NGUYÊN TẮC TRÍCH XUẤT:
1. CHỈ sử dụng thông tin có trong văn bản
2. ĐƯỢC PHÉP diễn giải để dễ hiểu, nhưng:
   - Giữ nguyên số liệu, mốc thời gian, điều kiện
   - Chỉ thay đổi cách diễn đạt
   - Không thay đổi phạm vi áp dụng
3. KHÔNG thêm điều kiện/hệ quả/kết luận mới
4. KHÔNG áp dụng logic suy luận ngoài văn bản
5. Nếu không có thông tin → nói rõ "không có quy định về vấn đề này"

HÌNH THỨC TRẢ LỜI:
- Văn bản thuần (không dùng Markdown)
- NGẮN GỌN, TRỰC TIẾP, CHỈ CẦN THIẾT
- Tối đa 1-2 câu
- KHÔNG thêm citations, metadata, hay thông tin bổ sung
- KHÔNG thêm "Cập nhật từ...", "năm không xác định"
- Chỉ nội dung chính của câu trả lời

NGỮ CẢNH (CÁC NGUỒN):
{context}

CÂU HỎI:
{question}

YÊU CẦU TRẢ LỜI:
- Trả lời trực tiếp câu hỏi
- Ngắn gọn, chính xác, không thêm thông tin thừa
- Không citations, không metadata

TRẢ LỜI:"""
    
    def _format_sources(self, source_info: List[Dict]) -> str:
        """Format source references for display - only cited sources (deduplicated)
        
        Args:
            source_info: Cited source information list (filtered) with original labels
            
        Returns:
            Formatted sources text
        """
        if not source_info:
            return ""
        
        # Deduplicate by title while preserving order and original label
        seen_titles = set()
        unique_sources = []
        for src in source_info:
            title = src.get('title', '')
            if title not in seen_titles:
                seen_titles.add(title)
                unique_sources.append(src)
        
        sources_text = "\n\nNGUỒN THAM KHẢO:"
        for src in unique_sources:
            label = src.get('label', '[Nguồn ?]')
            issue_date = src.get('issue_date', 'N/A')
            num_chunks = src.get('num_chunks', 1)
            
            sources_text += (
                f"\n{label} {src['title']} - {src['doc_type']} "
                f"({src['regulation_type']}) - Năm học: {src['academic_year']} - "
                f"Thời gian ban hành: {issue_date}"
            )
        
        return sources_text
    
    def _extract_cited_sources(self, answer: str, all_sources: List[Dict]) -> List[Dict]:
        """Extract only the sources that are actually cited in the answer
        
        Args:
            answer: Generated answer text
            all_sources: All available sources
            
        Returns:
            List of only cited sources, preserving order and deduplicating by title
        """
        cited_sources = []
        seen_titles = set()
        
        for src in all_sources:
            src_label = src.get('label', '')
            title = src.get('title', '')
            
            # Check if source is cited and not a duplicate
            if src_label and src_label in answer and title not in seen_titles:
                cited_sources.append(src)
                seen_titles.add(title)
        
        return cited_sources
    
    def print_retrieved_docs(self, documents: List[Document]) -> None:
        """Display retrieved documents for debugging
        
        Args:
            documents: Documents to display
        """
        pass