import re
import logging
from pathlib import Path
from typing import Dict, List
from langchain_core.documents import Document
from bs4 import BeautifulSoup
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

        # Extract issue date from content
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

        # Fallback to filename for date if not found in content
        if "issue_date" not in metadata:
            # Look for HK year in filename (e.g., HK232 -> 2024, HK222 -> 2023)
            hk_match = re.search(r"HK(\d{2})(\d)", filename, re.IGNORECASE)
            if hk_match:
                year_short, semester = hk_match.groups()
                # Semester 1: Sep-Jan, Semester 2: Feb-Jun, Semester 3 (Summer): Jul-Aug
                year = 2000 + int(year_short)
                if int(semester) >= 2:
                    year += 1
                metadata["issue_date"] = f"{year}-01-01" # Approximation

        # Add priority based on document type
        if any(keyword in filename.lower() for keyword in ["kết luận", "thông báo", "ket luan", "thong bao"]):
            metadata["priority"] = "high"
        else:
            metadata["priority"] = "normal"

        return metadata

    def _parse_html_tables(self, content: str) -> str:
        """Parse HTML tables in markdown content and convert to Key-Value text."""
        # Biểu thức chính quy tìm các thẻ <table>...</table>
        table_pattern = re.compile(r'<table.*?>.*?</table>', re.IGNORECASE | re.DOTALL)
        
        def table_replacer(match):
            table_html = match.group(0)
            soup = BeautifulSoup(table_html, 'html.parser')
            table = soup.find('table')
            if not table:
                return table_html
            
            # Xây dựng ma trận từ DOM để chứa dữ liệu un-merged
            rows = table.find_all('tr')
            if not rows: return table_html
            
            # Tính max columns
            max_cols = 0
            for row in rows:
                cols_count = sum(int(cell.get('colspan', 1)) for cell in row.find_all(['td', 'th']))
                max_cols = max(max_cols, cols_count)
            
            if max_cols == 0: return table_html
            
            # Khởi tạo ma trận trống
            matrix = [["" for _ in range(max_cols)] for _ in range(len(rows))]
            
            # Lấp đầy ma trận, xử lý rowspan/colspan
            for r_idx, row in enumerate(rows):
                c_idx = 0
                for cell in row.find_all(['td', 'th']):
                    # Tìm ô trống tiếp theo trên hàng hiện tại
                    while c_idx < max_cols and matrix[r_idx][c_idx] != "":
                        c_idx += 1
                        
                    if c_idx >= max_cols: break # Tránh lỗi index out of range
                    
                    rowspan = int(cell.get('rowspan', 1))
                    colspan = int(cell.get('colspan', 1))
                    text_val = cell.get_text(strip=True)
                    
                    # Điền giá trị vào ma trận cho tất cả các ô bị gộp
                    for r in range(rowspan):
                        for c in range(colspan):
                            if r_idx + r < len(rows) and c_idx + c < max_cols:
                                matrix[r_idx + r][c_idx + c] = text_val
                    
                    c_idx += colspan
            
            # Xây dựng câu Key-Value từ ma trận
            if not matrix: return table_html
            
            # Xác định số dòng dùng làm Header (các dòng chỉ chứa <th>)
            header_rows_count = 0
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if cells and all(c.name == 'th' for c in cells):
                    header_rows_count += 1
                else:
                    break
                    
            if header_rows_count == 0:
                header_rows_count = 1  # Mặc định dòng đầu tiên là header nếu không phân biệt được <th>
                
            # Xây dựng Composite Header (Gộp tiêu đề từ nhiều dòng header)
            headers = [""] * max_cols
            for r in range(header_rows_count):
                for c in range(max_cols):
                    val = matrix[r][c]
                    if val and val not in headers[c]:
                        headers[c] = (headers[c] + " " + val).strip()
            
            table_text = []
            
            for r_idx in range(header_rows_count, len(matrix)):
                row_data = matrix[r_idx]
                
                # Bỏ qua nếu dòng data giống hệt dòng tiêu đề đầu tiên (thường do table lặp header sau sang trang)
                if row_data == matrix[0]:
                    continue
                    
                row_parts = []
                for i in range(len(row_data)):
                    val = row_data[i]
                    if val and val != "None":
                        header_val = headers[i] if i < len(headers) else f"Column {i+1}"
                        if header_val.isdigit() or not header_val: # Bỏ qua header Vô nghĩa
                            row_parts.append(val)
                        else:
                            # Tránh lặp lại: Cột A: Cột A
                            if header_val == val:
                                row_parts.append(val)
                            else:
                                row_parts.append(f"{header_val}: {val}")
                
                if row_parts:
                    # Dùng set (lưu thứ tự qua dict) để loại bỏ các Key-Value trùng lặp liền kề sinh ra do gộp ô
                    unique_parts = list(dict.fromkeys(row_parts))
                    table_text.append(" - ".join(unique_parts))
            
            if not table_text:
                return table_html
                
            return "\n\n" + "\n".join(table_text) + "\n\n"
            
        return table_pattern.sub(table_replacer, content)
    
    def load_documents(self) -> List[Document]:
        """Load documents and convert HTML tables to text

        Returns:
            List of Documents
        """
        documents: List[Document] = []

        for md_file in self.base_path.rglob("*.md"):
            try:
                with open(md_file, "r", encoding="utf-8") as f:
                    content = f.read()

                # Parse HTML tables to Key-Value text
                content = self._parse_html_tables(content)

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

