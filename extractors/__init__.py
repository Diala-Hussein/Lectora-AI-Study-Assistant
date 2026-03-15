# Makes extractors a Python package
from .pdf_extractor import PDFExtractor
from .pptx_extractor import PPTXExtractor

__all__ = ['PDFExtractor', 'PPTXExtractor']