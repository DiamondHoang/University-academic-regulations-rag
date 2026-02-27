from typing import List, Dict, Optional
from datetime import datetime
import json
from pathlib import Path

from langchain_core.documents import Document


class ConversationMemory:
    """Manage conversation history and context with simple JSON persistence.

    Documents are stored as lightweight metadata so that the history can be
    reliably serialized to JSON.  The storage file is updated on every change.
    """

    def __init__(self, max_history: int = 5, persist_path: Optional[str] = None):
        """Initialize conversation memory

        Args:
            max_history: Maximum number of turns to keep
            persist_path: Optional path to JSON file for persistence
        """
        self.max_history = max_history
        self.history: List[Dict] = []
        # Default persistence file in the same folder as this module
        default_path = Path(__file__).parent / "history.json"
        self.persist_file = Path(persist_path) if persist_path else default_path
        self._load_history()

    def _doc_to_dict(self, doc: Document) -> Dict:
        """Convert a langchain Document into a JSON-friendly dict."""
        return {
            "page_content": doc.page_content,
            "metadata": doc.metadata,
        }

    def add_turn(self, question: str, answer: str, context_docs: Optional[List[Document]] = None) -> None:
        """Add a conversation turn and persist to disk

        Args:
            question: User question
            answer: Assistant answer
            context_docs: Optional context documents (will be serialized)
        """
        entry = {
            "question": question,
            "answer": answer,
            # convert documents to lightweight metadata if provided
            "context_docs": [self._doc_to_dict(d) for d in (context_docs or [])],
            "timestamp": datetime.now().isoformat()
        }
        self.history.append(entry)
        self._trim_history()
        self._save_history()

    def add_turn_with_data(self, turn_data: Dict) -> None:
        """Add a conversation turn with full data and persist

        Args:
            turn_data: Dictionary with question, answer, documents, analysis, confidence
        """
        turn = {
            "question": turn_data["question"],
            "answer": turn_data["answer"],
            # store only metadata of documents for serialization
            "context_docs": [self._doc_to_dict(d) for d in turn_data.get("documents", [])],
            "analysis": turn_data.get("analysis"),
            "confidence": turn_data.get("confidence"),
            "timestamp": datetime.now().isoformat()
        }
        self.history.append(turn)
        self._trim_history()
        self._save_history()

    def _trim_history(self) -> None:
        """Trim history to max length"""
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def get_context_string(self, include_last_n: int = 2) -> str:
        """Get formatted conversation history

        Args:
            include_last_n: Number of recent turns to include

        Returns:
            Formatted history string
        """
        if not self.history:
            return ""

        recent_history = self.history[-include_last_n:]
        context_parts = []

        for i, turn in enumerate(recent_history, 1):
            context_parts.append(f"Q{i}: {turn['question']}")
            context_parts.append(f"A{i}: {turn['answer']}")

        return "\n".join(context_parts)

    def get_last_turn(self) -> Optional[Dict]:
        """Get last conversation turn

        Returns:
            Last turn or None if empty
        """
        return self.history[-1] if self.history else None

    def get_history(self) -> List[Dict]:
        """Get full conversation history

        Returns:
            Copy of conversation history
        """
        return self.history.copy()

    def clear(self) -> None:
        """Clear conversation history and persist change"""
        self.history = []
        self._save_history()

    def _save_history(self) -> None:
        """Persist history to JSON file with JSON-friendly serialization."""
        def _json_default(o):
            # convert Document->dict; datetime->iso string; fallback to str
            from langchain_core.documents import Document
            if isinstance(o, Document):
                return {"page_content": o.page_content, "metadata": o.metadata}
            if isinstance(o, datetime):
                return o.isoformat()
            return str(o)

        try:
            self.persist_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.persist_file, "w", encoding="utf-8") as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2, default=_json_default)
        except Exception:
            # Fail silently; do not raise in production flow
            pass

    def _load_history(self) -> None:
        """Load history from JSON file if present"""
        try:
            if self.persist_file.exists():
                with open(self.persist_file, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                # Ensure timestamps are strings (ISO) and history is a list
                if isinstance(raw, list):
                    # we don't reconstruct Document objects; keep saved metadata
                    self.history = raw[-self.max_history:]
        except Exception:
            # On error, start with empty history
            self.history = []

