import re
from pathlib import Path
from typing import Dict, List
from langchain_core.documents import Document
from bs4 import BeautifulSoup
from config import Config


class RegulationDocumentLoader:
    """Loader for university regulation documents with metadata extraction and content cleaning."""
    
    def __init__(self, base_path: str = "md"):
        """Initialize loader
        
        Args:
            base_path: Base directory path for documents
        """
        self.base_path = Path(base_path)
    
    def extract_metadata_from_path(self, file_path: Path) -> Dict[str, str]:
        """Keep only the absolute file path as core metadata."""
        return {"file_path": str(file_path)}
    
    def extract_metadata_from_content(self, content: str, filename: str) -> Dict[str, str]:
        """Extract title and issue date from document content and filename.
        
        Args:
            content: Document text content
            filename: Name of the file (without extension)
            
        Returns:
            Dictionary with title and issue_date
        """
        metadata = {"title": filename}

        # OCR cleanup: fix spaced-out numbers (e.g., "2 0 2 4" -> "2024")
        content = re.sub(r"(\d)\s+(\d)", r"\1\2", content)

        # 1. Extract issue date from content using defined patterns
        for pattern in Config.DATE_PATTERNS:
            date_match = re.search(pattern, content, re.IGNORECASE)
            if date_match:
                try:
                    day, month, year = date_match.groups()
                    month = month.replace(" ", "")
                    # Normalize 2-digit years
                    if len(year) == 2:
                        year = "20" + year
                    metadata["issue_date"] = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                    break
                except (ValueError, IndexError):
                    continue

        # 2. Fallback: Parse date from filename patterns (e.g., HK232)
        if "issue_date" not in metadata:
            metadata["issue_date"] = self._infer_date_from_filename(filename)


        return metadata

    def _infer_date_from_filename(self, filename: str) -> str:
        """Infer an approximate issue date from semester codes in filename (e.g., HK232)."""
        hk_match = re.search(r"HK(\d{2})(\d)", filename, re.IGNORECASE)
        if hk_match:
            try:
                year_short, semester = hk_match.groups()
                year = 2000 + int(year_short)
                # Semester 2 and 3 usually fall in the next calendar year from the academic year start
                if int(semester) >= 2:
                    year += 1
                return f"{year}-01-01" # Approximation for sorting
            except (ValueError, IndexError):
                pass
        return "1970-01-01" # Default fallback for sorting

    def _clean_content(self, content: str) -> str:
        """Apply various cleaning operations to the document content."""
        # Remove page headers and metadata noise
        content = re.sub(Config.PAGE_HEADER_PATTERN, '', content, flags=re.MULTILINE)
        content = re.sub(Config.PAGE_INFO_PATTERN, '', content, flags=re.MULTILINE)
        
        # Parse HTML tables to searchable text
        content = self._parse_html_tables(content)
        
        return content.strip()

    def _parse_html_tables(self, content: str) -> str:
        """Find HTML tables in markdown and convert them to readable key-value sentences."""
        table_pattern = re.compile(r'<table.*?>.*?</table>', re.IGNORECASE | re.DOTALL)
        # Use regex sub with a lambda to keep the Markdown-as-string context intact
        return table_pattern.sub(lambda m: self._table_to_text(m.group(0)), content)

    def _table_to_text(self, html: str) -> str:
        """Main entry point: Convert a single HTML table to its text representation."""
        soup = BeautifulSoup(html, 'html.parser')
        table = soup.find('table')
        if not table: return ""

        rows = table.find_all('tr')
        if not rows: return ""

        # Step 1: Map out dimensions and fill a 2D matrix (handles colspan/rowspan)
        matrix = self._build_matrix(rows)
        if not matrix: return ""

        # Step 2: Extract and prioritize header labels
        headers, header_limit = self._extract_headers(rows, matrix)

        # Step 3: Convert data rows into searchable text sentences
        sentences = self._rows_to_sentences(matrix, headers, header_limit)

        return "\n\n" + "\n".join(sentences) + "\n\n" if sentences else ""

    def _build_matrix(self, rows) -> List[List[str]]:
        """Construct a de-merged 2D matrix of table cell contents."""
        max_cols = 0
        for row in rows:
            col_count = sum(int(c.get("colspan", 1)) for c in row.find_all(["td", "th"]))
            max_cols = max(max_cols, col_count)

        if max_cols == 0: return []

        matrix = [["" for _ in range(max_cols)] for _ in range(len(rows))]

        for r_idx, row in enumerate(rows):
            c_ptr = 0
            for cell in row.find_all(["td", "th"]):
                # Advance pointer past cells already filled by a previous rowspan
                while c_ptr < max_cols and matrix[r_idx][c_ptr]:
                    c_ptr += 1
                if c_ptr >= max_cols: break

                text = self._clean_text(cell.get_text(" ", strip=True))
                rs, cs = int(cell.get("rowspan", 1)), int(cell.get("colspan", 1))

                for dr in range(rs):
                    for dc in range(cs):
                        if r_idx + dr < len(rows) and c_ptr + dc < max_cols:
                            matrix[r_idx + dr][c_ptr + dc] = text
                c_ptr += cs
        return matrix

    def _extract_headers(self, rows, matrix) -> tuple:
        """Identify header labels from <th> tags or the first row of the matrix."""
        header_indices = []
        for i, row in enumerate(rows):
            if row.find_all("th"): header_indices.append(i)
            else: break # Stop at the first non-header (data) row

        # If no explicit <th> tags, fallback to treating the first row as the header
        header_limit = len(header_indices) if header_indices else 1
        max_cols = len(matrix[0])
        headers = [""] * max_cols

        for r in range(header_limit):
            for c in range(max_cols):
                val = matrix[r][c]
                # Combine stacked headers if they are unique
                if val and val not in headers[c]:
                    headers[c] = f"{headers[c]} {val}".strip()
        
        return headers, header_limit

    def _rows_to_sentences(self, matrix, headers, start_idx) -> List[str]:
        """Convert rows into 'Header: Value' strings for searchability."""
        results = []
        for r_idx in range(start_idx, len(matrix)):
            row = matrix[r_idx]
            if all(self._is_missing(cell) for cell in row):
                continue

            row_parts = []
            for c_idx, val in enumerate(row):
                if self._is_missing(val): continue
                
                hdr = headers[c_idx] if c_idx < len(headers) else ""
                # Avoid redundant mapping (e.g., "Date: 2024" instead of "2024: 2024")
                part = f"{hdr}: {val}" if hdr and hdr != val and not hdr.isdigit() else val
                row_parts.append(part)
            
            if row_parts:
                results.append(" - ".join(list(dict.fromkeys(row_parts))))
        return results

    def _is_missing(self, val: str) -> bool:
        """Check if a cell value represents empty or missing data."""
        return val.strip().lower() in {"", "none", "n/a", "-", "null", "no information"}

    def _clean_text(self, text: str) -> str:
        """Normalize whitespace in a string."""
        return " ".join(text.split())
    
    def load_documents(self) -> List[Document]:
        """Load all markdown documents from base_path, clean them, and extract metadata.

        Returns:
            List of LangChain Document objects.
        """
        documents: List[Document] = []

        if not self.base_path.exists():
            return []

        for md_file in self.base_path.rglob("*.md"):
            try:
                with open(md_file, "r", encoding="utf-8") as f:
                    raw_content = f.read()

                # Clean and transform content
                cleaned_content = self._clean_content(raw_content)

                # Build metadata from path and content
                metadata = self.extract_metadata_from_path(md_file)
                content_metadata = self.extract_metadata_from_content(
                    cleaned_content, md_file.stem
                )
                metadata.update(content_metadata)
                metadata["content_type"] = "markdown"

                documents.append(
                    Document(
                        page_content=cleaned_content,
                        metadata=metadata
                    )
                )

            except Exception as e:
                print(f"Error loading {md_file}: {e}")

        return documents

