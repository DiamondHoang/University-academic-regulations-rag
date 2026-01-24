from typing import List, Dict, Optional
from datetime import datetime

from langchain_core.documents import Document


class ConversationMemory:
    """Manage conversation history and context"""
    
    def __init__(self, max_history: int = 5):
        """Initialize conversation memory
        
        Args:
            max_history: Maximum number of turns to keep
        """
        self.max_history = max_history
        self.history: List[Dict] = []
    
    def add_turn(self, question: str, answer: str, context_docs: Optional[List[Document]] = None) -> None:
        """Add a conversation turn
        
        Args:
            question: User question
            answer: Assistant answer
            context_docs: Optional context documents
        """
        self.history.append({
            "question": question,
            "answer": answer,
            "context_docs": context_docs or [],
            "timestamp": datetime.now()
        })
        self._trim_history()
    
    def add_turn_with_data(self, turn_data: Dict) -> None:
        """Add a conversation turn with full data
        
        Args:
            turn_data: Dictionary with question, answer, documents, analysis, confidence
        """
        turn = {
            "question": turn_data["question"],
            "answer": turn_data["answer"],
            "context_docs": turn_data.get("documents", []),
            "analysis": turn_data.get("analysis"),
            "confidence": turn_data.get("confidence"),
            "timestamp": datetime.now()
        }
        self.history.append(turn)
        self._trim_history()
    
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
        """Clear conversation history"""
        self.history = []

