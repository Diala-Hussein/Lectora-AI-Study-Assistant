import pdfplumber
from typing import Optional
import io

class PDFExtractor:
    @staticmethod
    def extract(uploaded_file) -> Optional[str]:
        """Extract text from PDF file"""
        text = ""
        try:
            with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n--- Page {page_num} ---\n{page_text}\n"
            return text.strip()
        except Exception as e:
            print(f"PDF extraction error: {e}")
            return ""