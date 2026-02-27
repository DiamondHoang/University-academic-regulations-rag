import logging
from typing import List, Dict, Tuple
from langchain_core.documents import Document
from langchain_ollama import ChatOllama

from config import Config
from datetime import datetime

# Try to use dateutil for robust parsing, fall back to manual formats
try:
    from dateutil.parser import parse as _date_parse
except Exception:
    _date_parse = None

logger = logging.getLogger(__name__)


class ResponseGenerator:
    """Generate grounded responses from retrieved documents with confidence filtering

    To improve stability we maintain an in‑memory cache keyed by (query,context,analysis) so
    identical prompts return the same answer without calling the LLM again.  The LLM
    temperature should also be set to 0 (or very low) in configuration.
    """

    def __init__(self, llm: ChatOllama):
        self.llm = llm
        self.confidence_threshold = Config.CONFIDENCE_THRESHOLD
        # simple prompt->response cache to keep answers deterministic across runs
        self._cache: Dict[str, Dict] = {}

    def generate(
        self,
        query: str,
        documents: List[Document],
        analysis: Dict = None,
        conversation_history: str = "",
        clean_mode: bool = False,
    ) -> Dict:
        """Generate final response with confidence-filtered documents

        Returns dict: {"answer": str, "confidence": float, "sources": List, "selected_docs": int}
        """
        analysis = analysis or {}

        if not documents:
            # no context available when there are no docs, cache empty result keyed by query+analysis
            cache_key = f"{query}|||<no_context>|||{str(analysis)}"
            result = {
                "answer": self._fallback_no_answer(),
                "confidence": 0.0,
                "sources": [],
                "selected_docs": 0,
                "filtered_reason": "No documents retrieved"
            }
            self._cache[cache_key] = result
            return result

        try:
            # Filter documents by confidence threshold
            filtered_docs, selection_info = self._filter_by_confidence(documents)
            
            if not filtered_docs:
                return {
                    "answer": self._fallback_no_answer(),
                    "confidence": 0.0,
                    "sources": [],
                    "selected_docs": 0,
                    "filtered_reason": f"All {len(documents)} documents below confidence threshold ({self.confidence_threshold})"
                }

            # Build context from filtered documents
            context, sources_list = self._build_context(filtered_docs)

            # cache check now that context exists
            cache_key = f"{query}|||{context}|||{str(analysis)}"
            if cache_key in self._cache:
                logger.debug("returning cached response after context build")
                return self._cache[cache_key]

            prompt = self._build_prompt(
                query=query,
                context=context,
                analysis=analysis,
            )

            response = self.llm.invoke(prompt)
            answer = response.content.strip()

            # heuristic override: if model said no info but context clearly lists graduation modules,
            # construct a direct answer ourselves to avoid false negatives.
            if answer.startswith("Không tìm thấy") and "Các học phần tốt nghiệp" in context:
                import re
                m = re.search(r"Các học phần tốt nghiệp.*?\(([^)]+)\)", context)
                if m:
                    items = m.group(1)
                    answer = f"Học phần tốt nghiệp bao gồm các hình thức: {items}."

            # Filter answer to remove sentences not relevant to the query
            # This uses a generalized approach that works for any query type
            answer = self._filter_answer_by_relevance(answer, query, relevance_threshold=0.3)

            # Attach citations and build sources list
            answer_with_cites, sources = self._attach_citations(answer, sources_list)

            # If no cited sources were found, fallback to top filtered doc as source
            if not sources and filtered_docs:
                top = filtered_docs[0]
                sources = [{
                    "index": 1,
                    "title": top.metadata.get("title") or top.metadata.get("file_path", "unknown"),
                    "issue_date": top.metadata.get("issue_date", "N/A"),
                    "file_path": top.metadata.get("file_path", ""),
                    "confidence_score": top.metadata.get("confidence_score", 0.0),
                }]

            # Format final answer using concise template and attach source list
            formatted_answer = self._format_response(answer_with_cites, sources)

            # Calculate confidence based on filtered docs (simple heuristic)
            confidence = float(selection_info.get("avg_confidence", 0.0))

            result = {
                "answer": formatted_answer,
                "confidence": confidence,
                "sources": sources,
                "selected_docs": len(filtered_docs),
                "total_docs": len(documents),
                "selection_info": selection_info
            }
            # cache the response for future identical queries
            try:
                self._cache[cache_key] = result
            except Exception:
                logger.warning("Failed to cache response")
            return result

        except Exception as e:
            logger.error(f"Response generation failed: {e}", exc_info=True)
            return {
                "answer": self._fallback_error(),
                "confidence": 0.0,
                "sources": [],
                "selected_docs": 0
            }

    def _filter_by_confidence(self, documents: List[Document]) -> Tuple[List[Document], Dict]:
        """Filter documents by confidence threshold
        
        Args:
            documents: All retrieved documents
            
        Returns:
            Tuple of (filtered_documents, selection_info)
        """
        filtered_docs = []
        confidence_scores = []
        
        for doc in documents:
            conf_score = doc.metadata.get("confidence_score", 0.0)
            confidence_scores.append(conf_score)
            
            if conf_score >= self.confidence_threshold:
                filtered_docs.append(doc)
        
        selection_info = {
            "threshold": self.confidence_threshold,
            "selected": len(filtered_docs),
            "total": len(documents),
            "confidence_scores": confidence_scores[:len(documents)],
            "avg_confidence": sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        }
        
        logger.info(
            f"Document filtering: {len(filtered_docs)}/{len(documents)} selected "
            f"(threshold={self.confidence_threshold}, avg_conf={selection_info['avg_confidence']:.3f})"
        )
        # Ensure filtered docs are ordered by parsed issue date (newest first) then confidence
        from datetime import datetime as _dt
        def _ensure_parsed(d: Document):
            if "_parsed_issue_date" not in d.metadata:
                raw = d.metadata.get("issue_date") or ""
                if _date_parse:
                    try:
                        d.metadata["_parsed_issue_date"] = _date_parse(raw)
                    except Exception:
                        d.metadata["_parsed_issue_date"] = _dt.min
                else:
                    try:
                        d.metadata["_parsed_issue_date"] = _dt.strptime(raw, "%Y-%m-%d")
                    except Exception:
                        d.metadata["_parsed_issue_date"] = _dt.min

        for d in filtered_docs:
            _ensure_parsed(d)

        filtered_docs = sorted(
            filtered_docs,
            key=lambda x: (x.metadata.get("_parsed_issue_date", _dt.min), x.metadata.get("confidence_score", 0.0)),
            reverse=True,
        )

        return filtered_docs, selection_info

    # ======================================================
    # Context Builder
    # ======================================================

    def _build_context(self, docs: List[Document]) -> Tuple[str, List[Dict]]:
        """Build grounded context from filtered documents with clear source numbering.
        
        Deduplicates by file_path + title to prevent same source appearing twice.
        
        Returns:
            Tuple of (context_string, sources_list)
        """

        # Parse and attach a normalized datetime for each doc to allow
        # reliable sorting regardless of original date string format.
        def _parse_issue_date_raw(d: Document) -> datetime:
            raw = d.metadata.get("issue_date") or ""
            # Try dateutil if available
            if _date_parse:
                try:
                    return _date_parse(raw)
                except Exception:
                    pass

            # Try some common formats
            fmts = ["%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d", "%d-%m-%Y", "%Y-%m-%dT%H:%M:%S"]
            for f in fmts:
                try:
                    return datetime.strptime(raw, f)
                except Exception:
                    continue

            # If all fails, put very old sentinel so missing/invalid dates are treated as old
            try:
                return datetime.min
            except Exception:
                return datetime(1900, 1, 1)

        for d in docs:
            parsed = _parse_issue_date_raw(d)
            # store for later use
            d.metadata["_parsed_issue_date"] = parsed

        # Sort by parsed date (newest first) and confidence
        sorted_docs = sorted(
            docs,
            key=lambda x: (
                x.metadata.get("_parsed_issue_date", datetime.min),
                x.metadata.get("confidence_score", 0.0),
            ),
            reverse=True,
        )

        # Detect simple conflicts: same title + regulation_type but different dates
        conflict_groups: Dict[Tuple[str, str], List[Document]] = {}
        for d in sorted_docs:
            title = d.metadata.get("title", "")
            reg = d.metadata.get("regulation_type", "")
            key = (title, reg)
            conflict_groups.setdefault(key, []).append(d)

        # Build a set of documents that should be excluded because they are older
        excluded_old_docs = set()
        for key, group in conflict_groups.items():
            if len(group) > 1:
                # check if more than one distinct parsed date exist
                dates = {g.metadata.get("_parsed_issue_date") for g in group}
                if len(dates) > 1:
                    # keep only the newest (group already sorted by parsed date desc)
                    newest = sorted(group, key=lambda g: g.metadata.get("_parsed_issue_date", datetime.min), reverse=True)[0]
                    for g in group:
                        if g is not newest:
                            excluded_old_docs.add(id(g))

        # Group chunks by file_path + title to avoid duplicates from same source
        # Skip any older documents that were excluded due to conflicts above
        grouped_docs: Dict[str, List[Document]] = {}
        for doc in sorted_docs:
            if id(doc) in excluded_old_docs:
                # skip older conflicting versions so LLM only sees the newest source
                logger.info(f"Excluding older conflicting doc: {doc.metadata.get('title')} - {doc.metadata.get('issue_date')}")
                continue

            file_path = doc.metadata.get("file_path", "unknown")
            title = doc.metadata.get("title", "unknown")
            # Use file_path as primary dedup key - same file = same source
            dedup_key = file_path if file_path != "unknown" else title

            if dedup_key not in grouped_docs:
                grouped_docs[dedup_key] = []
            grouped_docs[dedup_key].append(doc)

        context_parts = []
        sources_list = []
        source_idx = 1

        # Iterate grouped_docs in insertion order so source numbering
        # preserves the order determined by `sorted_docs` (newest first).
        for dedup_key, group in grouped_docs.items():
            primary_doc = group[0]

            title = primary_doc.metadata.get("title", "unknown")
            doc_type = primary_doc.metadata.get("doc_type", "unknown")
            issue_date = primary_doc.metadata.get("issue_date", "N/A")
            regulation_type = primary_doc.metadata.get("regulation_type", "unknown")
            conf_score = primary_doc.metadata.get("confidence_score", 0.0)

            combined_content = "\n".join(doc.page_content for doc in group)

            # If this group represents a newer version replacing older ones, mark in header
            replaced_note = ""
            # find other docs with same title/regulation_type that were excluded
            title = primary_doc.metadata.get("title", "unknown")
            reg = primary_doc.metadata.get("regulation_type", "unknown")
            # if any excluded docs existed for this key, add a short note
            key = (title, reg)
            if any(id_doc in excluded_old_docs for id_doc in [id(x) for x in conflict_groups.get(key, [])]):
                replaced_note = "- NOTE: Phiên bản cũ đã được thay thế và bị loại khỏi ngữ cảnh.\n"

            header = (
                f"[Nguồn {source_idx}] {title}\n"
                f"- Loại: {doc_type}\n"
                f"- Nhóm: {regulation_type}\n"
                f"- Ngày ban hành: {issue_date}\n"
                f"{replaced_note}"
                f"- Độ tin cậy: {conf_score:.2%}\n"
            )

            context_parts.append(header + combined_content)
            
            # Add to sources list
            sources_list.append({
                "index": source_idx,
                "title": title,
                "issue_date": issue_date,
                "file_path": primary_doc.metadata.get("file_path", ""),
                "confidence_score": conf_score,
            })
            
            source_idx += 1

        return "\n\n".join(context_parts), sources_list

    # ======================================================
    # Prompt
    # ======================================================

    def _extract_query_topics(self, query: str) -> Dict[str, list]:
        """Extract key topics and keywords from user query
        
        Args:
            query: User question
            
        Returns:
            Dictionary with extracted topics
        """
        query_lower = query.lower()
        tokens = query_lower.split()
        
        # Remove common Vietnamese stop words
        stop_words = {"của", "là", "được", "có", "về", "cho", "với", "từ", "trong", 
                      "tại", "để", "nào", "gì", "không", "hay", "và", "hoặc", "như"}
        
        # Extract main keywords (remove stop words and short words)
        main_keywords = [t for t in tokens if len(t) > 2 and t not in stop_words]
        
        # Extract specific topics based on common patterns
        topics = {
            "main_keywords": main_keywords,
            "has_negation": any(kw in query_lower for kw in ["không", "chưa", "không được"]),
            "is_comparative": any(kw in query_lower for kw in ["tối đa", "tối thiểu", "nhiều hơn", "ít hơn", "cao hơn", "thấp hơn"]),
            "target_audience": self._extract_target_audience(query),
        }
        
        return topics
    
    def _extract_target_audience(self, query: str) -> str:
        """Extract target audience from query
        
        Args:
            query: User question
            
        Returns:
            Target audience string
        """
        query_lower = query.lower()
        
        if any(kw in query_lower for kw in ["sinh viên chính quy", "sinh viên"]):
            return "sinh_vien_chinh_quy"
        elif any(kw in query_lower for kw in ["cao học", "thạc sĩ", "master"]):
            return "hoc_vien_cao_hoc"
        elif any(kw in query_lower for kw in ["nghiên cứu sinh", "ncs", "phd", "ts"]):
            return "nghien_cuu_sinh"
        elif any(kw in query_lower for kw in ["vừa làm vừa học", "part-time", "parttime"]):
            return "sinh_vien_part_time"
        return "unknown"
    
    def _score_sentence_relevance(self, sentence: str, query_topics: Dict) -> float:
        """Score how relevant a sentence is to the query
        
        Args:
            sentence: A sentence from the answer
            query_topics: Extracted topics from query
            
        Returns:
            Relevance score between 0.0 and 1.0
        """
        sentence_lower = sentence.lower()
        keywords = query_topics.get("main_keywords", [])
        
        # Count how many keywords appear in this sentence
        matching_keywords = sum(1 for kw in keywords if kw in sentence_lower)
        
        if not keywords:
            return 1.0  # If no keywords, keep everything
        
        # Relevance = percentage of keywords that appear in sentence
        relevance = matching_keywords / len(keywords) if keywords else 0.5
        
        # Penalty if sentence has very different topic
        if self._is_off_topic_sentence(sentence_lower, query_topics):
            relevance *= 0.3
        
        return max(0.0, min(1.0, relevance))
    
    def _is_off_topic_sentence(self, sentence: str, query_topics: Dict) -> bool:
        """Check if a sentence is off-topic
        
        Args:
            sentence: Sentence text (lowercase)
            query_topics: Extracted topics from query
            
        Returns:
            True if sentence is off-topic
        """
        # Specific patterns that are usually off-topic when not explicitly asked
        off_topic_patterns = [
            ("chuẩn đầu ra tối thiểu" if "chuẩn đầu ra" not in str(query_topics) else None),
            ("chuẩn đầu ra tối đa" if "chuẩn đầu ra" not in str(query_topics) else None),
        ]
        
        off_topic_patterns = [p for p in off_topic_patterns if p]  # Remove None
        
        return any(pattern in sentence for pattern in off_topic_patterns)

    def _build_prompt(
        self,
        query: str,
        context: str,
        analysis: Dict,
    ) -> str:
        """Build prompt optimized for legal grounding with inline citations.
        
        Uses generic principles that work for all query types.
        """

        target = analysis.get("target_audience", "unknown")

        return f"""Bạn là chuyên gia tư vấn quy định đào tạo của trường đại học.

    HƯỚNG DẪN NGẮN:
    - Chỉ sử dụng thông tin có trong phần CONTEXT bên dưới. KHÔNG thêm thông tin hoặc suy đoán từ nguồn khác.
    - Nếu sau khi đọc kỹ CONTEXT vẫn không thể trả lời, dùng mẫu: "Không tìm thấy quy định về <vấn đề> trong các văn bản được cung cấp." (thay <vấn đề> bằng nội dung câu hỏi).

    CÁCH TRÍCH DẪN:
    - Dùng inline citation [Nguồn N] ngay trước dấu chấm của câu/ý tương ứng. N là số thứ tự nguồn trong CONTEXT.
    - Nếu hai nguồn mâu thuẫn, trích dẫn nguồn MỚI NHẤT và ghi kèm: "(Quy định đã thay đổi so với [Nguồn X])."

    YÊU CẦU:
    - KHÔNG ĐƯỢC TRẢ LỜI BẰNG ĐỊNH DẠNG MARKDOWN.
    - Trả lời trực tiếp, ngắn gọn (tối đa 3 câu) định dạng plain text.
    - Giữ nguyên số, điều kiện và thời hạn như trong văn bản.
    - Không thêm ví dụ, giải thích dư thừa hoặc suy luận pháp lý.

    ĐỐI TƯỢNG: {target}

    CÂU HỎI:
    {query}

    CONTEXT:
    {context}

    TRẢ LỜI:
    """

    # ======================================================
    # Fallback
    # ======================================================

    def _fallback_no_answer(self) -> str:
        return (
            "Không tìm thấy thông tin phù hợp trong các văn bản hiện có. "
            "Bạn có thể cung cấp thêm chi tiết hoặc hỏi rõ hơn."
        )

    def _fallback_error(self) -> str:
        return "Đã xảy ra lỗi khi tạo câu trả lời. Vui lòng thử lại sau."

    def _filter_answer_by_relevance(
        self, 
        answer: str, 
        query: str,
        relevance_threshold: float = 0.3
    ) -> str:
        """Filter answer to remove sentences not relevant to the query
        
        This is a generalized filtering approach that works for any query type.
        It extracts query topics and scores sentence relevance.
        
        Args:
            answer: Generated answer
            query: User question
            relevance_threshold: Minimum relevance score to keep sentence (0.0-1.0)
            
        Returns:
            Filtered answer with only relevant sentences
        """
        import re
        
        # Extract topics from query
        query_topics = self._extract_query_topics(query)
        
        # Split answer into sentences
        # Handle Vietnamese sentence endings: . ! ? followed by space or end
        sentences = re.split(r'(?<=[.!?])\s+', answer)
        
        filtered_sentences = []
        
        for sent in sentences:
            if not sent.strip():
                continue
            
            # Score relevance of this sentence
            relevance = self._score_sentence_relevance(sent, query_topics)
            
            if relevance >= relevance_threshold:
                filtered_sentences.append(sent)
                logger.debug(f"KEPT sentence (score={relevance:.2f}): {sent[:50]}...")
            else:
                logger.debug(f"FILTERED out (score={relevance:.2f}): {sent[:50]}...")
        
        # Rejoin sentences
        filtered_answer = " ".join(filtered_sentences).strip()
        
        # If all sentences were filtered, return original answer
        if not filtered_answer:
            logger.warning("All sentences were filtered out, returning original answer")
            return answer
        
        return filtered_answer

    def _attach_citations(self, answer: str, sources_list: List[Dict]):
        """Extract citations from answer and build reference list.
        
        Validates and remaps invalid citations to prevent citing non-existent sources.
        Returns: (answer_with_fixed_citations, cited_sources)
        """
        import re

        # Get valid source indices
        valid_indices = {src.get("index") for src in sources_list}
        
        # Extract citations from answer [Nguồn X] format
        citation_pattern = r'\[Nguồn\s+(\d+)\]'
        matches = re.findall(citation_pattern, answer)
        used_indices = set(int(m) for m in matches)
        
        # Check for invalid citations
        invalid_indices = used_indices - valid_indices
        
        # If there are invalid citations, log and remap them
        fixed_answer = answer
        if invalid_indices:
            logger.warning(
                f"Found citations to non-existent sources: {invalid_indices}. "
                f"Valid sources are: {valid_indices}"
            )
            # Remap invalid citations to the first valid source
            if valid_indices:
                first_valid = min(valid_indices)
                for invalid_idx in sorted(invalid_indices):
                    pattern = f"\\[Nguồn\\s+{invalid_idx}\\]"
                    fixed_answer = re.sub(pattern, f"[Nguồn {first_valid}]", fixed_answer)
                    logger.info(f"Remapped [Nguồn {invalid_idx}] to [Nguồn {first_valid}]")
        
        # Extract citations from fixed answer
        matches = re.findall(citation_pattern, fixed_answer)
        used_indices = set(int(m) for m in matches)

        # Build cited sources in order based on indices
        cited_sources = []
        for idx in sorted(used_indices):
            for src in sources_list:
                if src.get("index") == idx:
                    cited_sources.append(src)
                    break

        # --- additional step: renumber sources sequentially so that
        # citations start at 1 and have no gaps.  This fixes cases where
        # only one source remains but it was originally labelled 2, etc.
        if cited_sources:
            mapping = {}
            for new_idx, src in enumerate(cited_sources, start=1):
                old_idx = src.get("index")
                if old_idx != new_idx:
                    mapping[old_idx] = new_idx
                    src["index"] = new_idx
            if mapping:
                # replace in answer text
                for old, new in mapping.items():
                    fixed_answer = re.sub(rf"\[Nguồn\s+{old}\]", f"[Nguồn {new}]", fixed_answer)
                logger.info(f"Reindexed cited sources to remove gaps: {mapping}")

        return fixed_answer, cited_sources

    def _format_response(self, raw_answer: str, sources: List[Dict]) -> str:
        """Format the raw answer and append sources with proper formatting.
        
        Appends "Nguồn:" section at the end with source indices.
        """
        if not raw_answer:
            summary = "Không tìm thấy thông tin liên quan trong các văn bản cung cấp."
        else:
            summary = raw_answer.strip()

        # Append sources section
        out_lines = [summary]
        
        if not sources:
            out_lines.append("\nNguồn: Không tìm thấy nguồn trong các văn bản đã cung cấp.")
        else:
            out_lines.append("\nNguồn:")
            for s in sources:
                idx = s.get("index", "?")
                conf = s.get("confidence_score", 0.0)
                issue = s.get("issue_date", "N/A")
                title = s.get("title", s.get("file_path", "unknown"))
                out_lines.append(f"[{idx}] {title} — {issue}")

        return "\n".join(out_lines)

