"""
core/text_processor.py
----------------------
Cleans raw text extracted from PDF/PPTX slides and splits it into
semantically meaningful chunks (2-4 sentences) suitable for embedding.

Improvements over the original:
  - Slide-artifact removal (page numbers, footers, headers)
  - Bullet-point merging into full sentences
  - Heading detection and tagging
  - Overlapping context windows for chunk creation
  - Returns both raw sentences and chunk objects
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """A context window containing 2–4 sentences from a slide."""
    text: str
    sentences: List[str]
    start_idx: int          # index of first sentence in global sentence list
    end_idx: int            # index of last sentence (inclusive)
    heading: Optional[str] = None


# ---------------------------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------------------------

_SLIDE_ARTIFACT_PATTERNS = [
    re.compile(r"^\s*\d+\s*$"),                        # lone page numbers
    re.compile(r"(?i)^\s*(slide|page)\s*\d+\s*$"),     # "Slide 3"
    re.compile(r"(?i)^(confidential|draft|copyright)"), # footers
    re.compile(r"^\s*[-–—]+\s*$"),                      # separator lines
    re.compile(r"^\s*\|\s*$"),                          # lone pipe chars
    re.compile(r"^https?://\S+$"),                      # bare URLs
]

_HEADING_PATTERN = re.compile(
    r"^(?:[A-Z][A-Z\s\d:,\-]{2,60}|[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,5})$"
)


def _is_artifact(line: str) -> bool:
    return any(p.match(line.strip()) for p in _SLIDE_ARTIFACT_PATTERNS)


def _is_heading(line: str) -> bool:
    stripped = line.strip()
    if len(stripped) < 4 or len(stripped) > 80:
        return False
    if stripped.endswith("."):
        return False
    return bool(_HEADING_PATTERN.match(stripped))


def _merge_bullet_lines(lines: List[str]) -> List[str]:
    """
    Merge bullet-point fragments into single sentences.
    Lines starting with -, *, •, · or a digit+dot are treated as bullets.
    Consecutive bullets under the same heading are joined with '; '.
    """
    bullet_re = re.compile(r"^\s*[-*•·]|\s*^\d+[.)]\s+")
    merged: List[str] = []
    buffer: List[str] = []

    def flush():
        if buffer:
            merged.append(" ".join(b.strip() for b in buffer))
            buffer.clear()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            flush()
            continue
        if bullet_re.match(line):
            text = bullet_re.sub("", line).strip()
            if text:
                buffer.append(text)
        else:
            flush()
            merged.append(stripped)

    flush()
    return merged


def clean_text(raw_text: str) -> Tuple[str, List[str]]:
    """
    Clean raw slide text.

    Parameters
    ----------
    raw_text : str – raw text from PDF/PPTX extractor

    Returns
    -------
    (cleaned_str, headings_list)
        cleaned_str   : cleaned, normalised text
        headings_list : list of detected heading strings
    """
    lines = raw_text.splitlines()
    headings: List[str] = []
    kept: List[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if _is_artifact(stripped):
            continue
        if _is_heading(stripped):
            headings.append(stripped)
            kept.append(stripped + ".")   # treat heading as a sentence seed
            continue
        kept.append(stripped)

    merged = _merge_bullet_lines(kept)

    # normalise whitespace and fix common slide quirks
    cleaned_lines = []
    for line in merged:
        line = re.sub(r"\s+", " ", line).strip()
        # ensure terminal punctuation so sentence splitter works
        if line and line[-1] not in ".!?:":
            line += "."
        if line:
            cleaned_lines.append(line)

    cleaned_str = " ".join(cleaned_lines)
    return cleaned_str, headings


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

_SENT_BOUNDARY = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z])"
    r"|(?<=[.!?])\s*\n"
    r"|(?<=\.)\s{2,}"
)


def split_sentences(text: str) -> List[str]:
    """
    Split cleaned text into individual sentences.
    Filters out sentences that are too short to be meaningful.
    """
    raw_sents = _SENT_BOUNDARY.split(text)
    sentences: List[str] = []
    for s in raw_sents:
        s = s.strip()
        word_count = len(s.split())
        if word_count >= 4:
            sentences.append(s)
    return sentences


# ---------------------------------------------------------------------------
# Chunking with overlap
# ---------------------------------------------------------------------------

def build_chunks(
    sentences: List[str],
    chunk_size: int = 3,
    overlap: int = 1,
    headings: Optional[List[str]] = None,
) -> List[Chunk]:
    """
    Build overlapping context chunks from a sentence list.

    Parameters
    ----------
    sentences  : list of cleaned sentences
    chunk_size : target number of sentences per chunk (2–4)
    overlap    : number of sentences shared between adjacent chunks
    headings   : optional list of heading strings for tagging

    Returns
    -------
    List[Chunk]
    """
    chunk_size = max(2, min(chunk_size, 4))
    overlap = max(0, min(overlap, chunk_size - 1))
    heading_set = set(headings or [])

    chunks: List[Chunk] = []
    step = chunk_size - overlap
    n = len(sentences)

    i = 0
    while i < n:
        end = min(i + chunk_size, n)
        window = sentences[i:end]

        # detect heading tag for this chunk
        chunk_heading: Optional[str] = None
        for sent in window:
            base = sent.rstrip(".")
            if base in heading_set:
                chunk_heading = base
                break

        chunk_text = " ".join(window)
        chunks.append(
            Chunk(
                text=chunk_text,
                sentences=window,
                start_idx=i,
                end_idx=end - 1,
                heading=chunk_heading,
            )
        )
        i += step
        if end == n:
            break

    return chunks


# ---------------------------------------------------------------------------
# Top-level pipeline entry point
# ---------------------------------------------------------------------------

def process_lecture_text(
    raw_text: str,
    chunk_size: int = 3,
    overlap: int = 1,
) -> Tuple[List[str], List[Chunk], List[str]]:
    """
    Full processing pipeline: clean → split → chunk.

    Parameters
    ----------
    raw_text   : raw text from extractor
    chunk_size : sentences per chunk
    overlap    : overlap between chunks

    Returns
    -------
    (sentences, chunks, headings)
    """
    cleaned, headings = clean_text(raw_text)
    sentences = split_sentences(cleaned)
    chunks = build_chunks(sentences, chunk_size=chunk_size, overlap=overlap, headings=headings)
    return sentences, chunks, headings


# ---------------------------------------------------------------------------
# Backward-compatibility shim
# ---------------------------------------------------------------------------

class LectureTextProcessor:
    """
    Drop-in replacement for the original LectureTextProcessor class.

    Wraps the new functional API so existing code that does:
        processor = LectureTextProcessor()
        cleaned   = processor.clean_slide_text(raw_text)
        sentences = processor.preprocess_sentences(cleaned_text)
    continues to work without modification.

    preprocess_sentences() returns List[Tuple[str, float]] where the float
    is a simple importance score, matching the original contract expected by
    app.py ( `for s, _ in assistant.sentences` ).
    """

    def clean_slide_text(self, raw_text: str) -> str:
        """Clean raw slide text and return a single normalised string."""
        cleaned, _ = clean_text(raw_text)
        return cleaned

    def preprocess_sentences(self, cleaned_text: str) -> List[Tuple[str, float]]:
        """
        Split cleaned text into sentences and attach a basic importance score.

        Returns
        -------
        List[Tuple[str, float]]
            Each element is (sentence_text, importance_score).
            Scores are in [0, 1] based on length and numeric content.
        """
        from utils.similarity import sentence_importance_scores
        sentences = split_sentences(cleaned_text)
        if not sentences:
            return []
        scores = sentence_importance_scores(sentences)
        return list(zip(sentences, scores.tolist()))
