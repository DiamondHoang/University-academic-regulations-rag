import re
import logging
from pathlib import Path
from typing import Dict, List
from langchain_core.documents import Document
from config import Config

logger = logging.getLogger(__name__)


class RegulationDocumentLoader:
    """Load and extract metadata from regulation documents"""
    
    def __init__(self, base_path: str = "md"):
        """Initialize loader
        
        Args:
            base_path: Base directory path for documents
        """
        self.base_path = Path(base_path)
    
    def extract_metadata_from_path(self, file_path: Path) -> Dict[str, str]:
        """Extract metadata from file path structure
        
        Args:
            file_path: Path to document file
            
        Returns:
            Dictionary with extracted metadata
        """
        parts = file_path.relative_to(self.base_path).parts
        filename = file_path.stem
        
        metadata = {
            "file_path": str(file_path),
            "doc_type": parts[0] if len(parts) > 0 else "unknown", # DTDH
            "regulation_type": parts[1] if len(parts) > 1 else "unknown", # QDHV
        }
        
        return metadata
    
    def extract_metadata_from_content(self, content: str, filename: str) -> Dict[str, str]:
        metadata = {"title": filename}

        # OCR cleaning
        content = re.sub(r"(\d)\s+(\d)", r"\1\2", content)

        # Extract issue date
        for pattern in Config.DATE_PATTERNS:
            date_match = re.search(pattern, content, re.IGNORECASE)
            if date_match:
                try:
                    day, month, year = date_match.groups()

                    month = month.replace(" ", "")

                    if len(year) == 2:
                        year = "20" + year

                    metadata["issue_date"] = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                    break
                except Exception:
                    continue

        return metadata
    
    def load_documents(self) -> List[Document]:
        """Load documents without table conversion

        Returns:
            List of Documents
        """
        documents: List[Document] = []

        for md_file in self.base_path.rglob("*.md"):
            try:
                with open(md_file, "r", encoding="utf-8") as f:
                    content = f.read()

                metadata = self.extract_metadata_from_path(md_file)
                content_metadata = self.extract_metadata_from_content(
                    content, md_file.stem
                )
                metadata.update(content_metadata)

                documents.append(
                    Document(
                        page_content=content.strip(),
                        metadata={
                            **metadata,
                            "content_type": "markdown"
                        }
                    )
                )

            except Exception as e:
                logger.error(
                    f"Error loading {md_file}: {e}",
                    exc_info=True
                )

        logger.info(
            f"Loaded {len(documents)} documents from {self.base_path}"
        )
        return documents

