from typing import List, Dict, Optional, Any
from datetime import datetime
import json
from pathlib import Path

from langchain_core.documents import Document

class ConversationMemory:
    """Manage conversation history and context with JSON persistence."""

    def __init__(self, max_history: int = 5, session_id: Optional[str] = None, disable_persistence: bool = False):
        """Initialize conversation memory
        
        Args:
            max_history: Max number of turns to keep
            session_id: Unique ID for file-based history
            disable_persistence: If True, keep history in RAM only.
        """
        self.max_history = max_history
        self.history: List[Dict[str, Any]] = []
        self.disable_persistence = disable_persistence
        
        if disable_persistence:
            self.persist_file = None
        else:
            filename = f"history_{session_id}.json" if session_id else "history.json"
            self.persist_file = Path(__file__).parent / filename
            self._load_history()

    def _doc_to_dict(self, doc: Document) -> Dict[str, Any]:
        """Serialize a Document into a JSON-friendly dict."""
        return {
            "page_content": doc.page_content,
            "metadata": doc.metadata,
        }

    def add_turn(self, question: str, answer: str, context_docs: Optional[List[Document]] = None) -> None:
        """Add a turn to history and persist."""
        entry = {
            "question": question,
            "answer": answer,
            "context_docs": [self._doc_to_dict(d) for d in (context_docs or [])],
            "timestamp": datetime.now().isoformat()
        }
        self.history.append(entry)
        self._trim_history()
        self._save_history()

    def add_turn_with_data(self, turn_data: Dict[str, Any]) -> None:
        """Add a complete turn with analysis data."""
        turn = {
            "question": turn_data["question"],
            "answer": turn_data["answer"],
            "context_docs": [self._doc_to_dict(d) for d in turn_data.get("documents", [])],
            "analysis": turn_data.get("analysis"),
            "confidence": turn_data.get("confidence"),
            "timestamp": datetime.now().isoformat()
        }
        self.history.append(turn)
        self._trim_history()
        self._save_history()

    def _trim_history(self) -> None:
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def get_context_string(self, include_last_n: int = 2) -> str:
        """Format history for LLM context prompts."""
        if not self.history:
            return ""
        
        recent = self.history[-include_last_n:]
        return "\n".join([f"Q: {t['question']}\nA: {t['answer']}" for t in recent])

    def get_history_messages(self, include_last_n: int = 3) -> List[Dict[str, str]]:
        """Return history as structured OpenAI-style messages."""
        messages = []
        for turn in self.history[-include_last_n:]:
            messages.append({"role": "user", "content": turn["question"]})
            messages.append({"role": "assistant", "content": turn["answer"]})
        return messages

    def clear(self) -> None:
        self.history = []
        self._save_history()

    def _save_history(self) -> None:
        if self.disable_persistence or not self.persist_file:
            return

        def _json_default(o):
            if isinstance(o, Document):
                return self._doc_to_dict(o)
            if isinstance(o, datetime):
                return o.isoformat()
            return str(o)

        try:
            self.persist_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.persist_file, "w", encoding="utf-8") as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2, default=_json_default)
        except Exception as e:
            print(f"[Memory] Failed to save history: {e}")

    def _load_history(self) -> None:
        if not self.persist_file or not self.persist_file.exists():
            return

        try:
            with open(self.persist_file, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, list):
                self.history = raw[-self.max_history:]
        except Exception as e:
            print(f"[Memory] Failed to load history: {e}")
            self.history = []
