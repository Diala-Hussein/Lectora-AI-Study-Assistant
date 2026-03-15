"""
core/qa_engine.py
-----------------
OpenAI-powered QA engine with local semantic pre-filtering.

Cost strategy (minimal token usage):
  1. Split lecture into chunks locally (no API call).
  2. Use lightweight local TF-IDF keyword scoring to pre-filter
     down to the TOP 3 most relevant chunks (~300-500 tokens).
  3. Send ONLY those chunks + the question to gpt-4o-mini.
  4. gpt-4o-mini reads the context and writes a real, grounded answer.

This means a typical Q&A call costs ~$0.0002 - $0.0005 (fraction of a cent).
No embeddings API called — saves money and latency.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from typing import List, Optional, Tuple

from core.text_processor import clean_text, split_sentences, build_chunks, Chunk, LectureTextProcessor
from utils.openai_client import chat, count_tokens

# ── tuneable constants ──────────────────────────────────────────────────────
MAX_CONTEXT_TOKENS  = 1800   # hard cap on lecture context sent to API
TOP_K_CHUNKS        = 4      # pre-filter: keep this many chunks before API call
CHUNK_SIZE          = 4      # sentences per chunk
OVERLAP             = 1      # overlap between chunks


# ---------------------------------------------------------------------------
# Local keyword scorer (zero API cost)
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would could should may might shall can this that these those "
    "i me my we our you your he she it its they them their what which "
    "who whom when where why how all each every both and or in on at to "
    "for of with by about into through before after above below from up "
    "down out off over under again further then once there here".split()
)

def _tokenize(text: str) -> List[str]:
    return [w for w in re.findall(r"\b[a-z]{2,}\b", text.lower())
            if w not in _STOPWORDS]

def _score_chunk(query_tokens: List[str], chunk_text: str) -> float:
    chunk_tokens = Counter(_tokenize(chunk_text))
    return sum(chunk_tokens.get(t, 0) for t in query_tokens)


# ---------------------------------------------------------------------------
# QA Engine
# ---------------------------------------------------------------------------

class QAEngine:
    """
    Answers questions about lecture content using OpenAI gpt-4o-mini.
    Locally pre-filters chunks to minimise token spend.
    """

    def __init__(self):
        self._chunks    : List[Chunk] = []
        self._sentences : List[str]   = []
        self._raw_text  : str         = ""

    # ── indexing ─────────────────────────────────────────────────────────────

    def index(self, raw_text: str) -> None:
        """Process and store lecture chunks. Pure local — no API call."""
        if raw_text == self._raw_text:
            return
        cleaned, headings   = clean_text(raw_text)
        self._sentences     = split_sentences(cleaned)
        self._chunks        = build_chunks(
            self._sentences,
            chunk_size=CHUNK_SIZE,
            overlap=OVERLAP,
            headings=headings,
        )
        self._raw_text = raw_text

    def fit(self, sentence_texts: List[str]) -> None:
        """app.py compatibility: accepts List[str] from LectureTextProcessor."""
        self.index(" ".join(sentence_texts))

    # ── retrieval (local, free) ───────────────────────────────────────────────

    def _retrieve_local(self, question: str, top_k: int = TOP_K_CHUNKS) -> List[Chunk]:
        """Rank chunks by keyword overlap and return the top-k."""
        q_tokens = _tokenize(question)
        scored = sorted(
            self._chunks,
            key=lambda c: _score_chunk(q_tokens, c.text),
            reverse=True,
        )
        return scored[:top_k]

    # ── answer (one API call) ─────────────────────────────────────────────────

    def answer(self, question: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Answer `question` using lecture content.

        Returns
        -------
        List[Tuple[str, float]]
            [(answer_text, confidence_score)]  — single item list so app.py
            loop works unchanged.
        """
        if not self._chunks:
            return [("No lecture has been processed yet.", 0.0)]

        # 1. local pre-filter — free
        best_chunks = self._retrieve_local(question, top_k=TOP_K_CHUNKS)

        # 2. build context string, respecting token budget
        context_parts = []
        total = 0
        for chunk in best_chunks:
            tokens = count_tokens(chunk.text)
            if total + tokens > MAX_CONTEXT_TOKENS:
                break
            context_parts.append(chunk.text)
            total += tokens

        context = "\n\n".join(context_parts)

        # 3. single API call to gpt-4o-mini
        system = (
            "You are a study assistant. Answer the student's question using "
            "ONLY the lecture excerpts provided. Be clear and concise. "
            "If the answer is not in the excerpts, say so honestly."
        )
        user = f"LECTURE EXCERPTS:\n{context}\n\nQUESTION: {question}"

        answer_text = chat(system=system, user=user, max_tokens=350)
        return [(answer_text, 1.0)]
