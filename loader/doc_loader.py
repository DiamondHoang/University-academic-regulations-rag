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
        """Extract metadata from file path structure (hierarchy: Audience/Category/File).
        
        Args:
            file_path: Path to document file
            
        Returns:
            Dictionary with 'doc_type' and 'regulation_type'
        """
        try:
            relative_path = file_path.relative_to(self.base_path)
            parts = relative_path.parts
            
            return {
                "file_path": str(file_path),
                "doc_type": parts[0] if len(parts) > 0 else "unknown",
                "regulation_type": parts[1] if len(parts) > 1 else "unknown",
            }
        except ValueError:
            # Handle cases where file is outside base_path
            return {
                "file_path": str(file_path),
                "doc_type": "unknown",
                "regulation_type": "unknown",
            }
    
    def extract_metadata_from_content(self, content: str, filename: str) -> Dict[str, str]:
        """Extract title, issue date, and priority from document content and filename.
        
        Args:
            content: Document text content
            filename: Name of the file (without extension)
            
        Returns:
            Dictionary with title, issue_date, and priority
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

        # 3. Determine priority based on keywords
        high_priority_keywords = ["kết luận", "thông báo", "ket luan", "thong bao"]
        if any(kw in filename.lower() for kw in high_priority_keywords):
            metadata["priority"] = "high"
        else:
            metadata["priority"] = "normal"

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
        
        def table_replacer(match):
            table_html = match.group(0)
            soup = BeautifulSoup(table_html, 'html.parser')
            table = soup.find('table')
            if not table:
                return table_html
            
            rows = table.find_all('tr')
            if not rows: return table_html
            
            # 1. Determine table dimensions
            max_cols = 0
            for row in rows:
                cols_count = sum(int(cell.get('colspan', 1)) for cell in row.find_all(['td', 'th']))
                max_cols = max(max_cols, cols_count)
            
            if max_cols == 0: return table_html
            
            # 2. Build de-merged matrix
            matrix = [["" for _ in range(max_cols)] for _ in range(len(rows))]
            for r_idx, row in enumerate(rows):
                c_idx = 0
                for cell in row.find_all(['td', 'th']):
                    while c_idx < max_cols and matrix[r_idx][c_idx] != "":
                        c_idx += 1
                    if c_idx >= max_cols: break
                    
                    rowspan = int(cell.get('rowspan', 1))
                    colspan = int(cell.get('colspan', 1))
                    text_val = cell.get_text(strip=True)
                    
                    for r in range(rowspan):
                        for c in range(colspan):
                            if r_idx + r < len(rows) and c_idx + c < max_cols:
                                matrix[r_idx + r][c_idx + c] = text_val
                    c_idx += colspan
            
            # 3. Identify header rows (all cells are <th>)
            header_rows_count = 0
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if cells and all(c.name == 'th' for c in cells):
                    header_rows_count += 1
                else: break
            
            if header_rows_count == 0: header_rows_count = 1
                
            # 4. Build composite headers
            headers = [""] * max_cols
            for r in range(header_rows_count):
                for c in range(max_cols):
                    val = matrix[r][c]
                    if val and val not in headers[c]:
                        headers[c] = (headers[c] + " " + val).strip()
            
            # 5. Build key-value representation for each data row
            table_text = []
            for r_idx in range(header_rows_count, len(matrix)):
                row_data = matrix[r_idx]
                if row_data == matrix[0]: continue # Skip repeated headers
                    
                row_parts = []
                for i, val in enumerate(row_data):
                    if val and val != "None":
                        header_val = headers[i] if i < len(headers) else f"Column {i+1}"
                        if header_val.isdigit() or not header_val or header_val == val:
                            row_parts.append(val)
                        else:
                            row_parts.append(f"{header_val}: {val}")
                
                if row_parts:
                    # Deduplicate adjacent identical parts (caused by cell merging)
                    unique_parts = list(dict.fromkeys(row_parts))
                    table_text.append(" - ".join(unique_parts))
            
            return "\n\n" + "\n".join(table_text) + "\n\n" if table_text else table_html
            
        return table_pattern.sub(table_replacer, content)
    
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
                pass

        return documents

