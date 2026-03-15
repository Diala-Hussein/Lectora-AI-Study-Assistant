"""
core/__init__.py
----------------
Package initialiser.  Imports are intentionally lazy so that a missing
optional dependency (e.g. sentence-transformers) does not crash the whole
app at startup.
"""

# Expose the compatibility class names that app.py imports directly
try:
    from .text_processor import LectureTextProcessor  # noqa: F401
except Exception:
    pass

try:
    from .qa_engine import QAEngine  # noqa: F401
except Exception:
    pass

try:
    from .summarizer import TextRankSummarizer  # noqa: F401
except Exception:
    pass

try:
    from .mcq_generator import MCQGenerator  # noqa: F401
except Exception:
    pass
