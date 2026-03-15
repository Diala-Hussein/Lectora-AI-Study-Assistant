"""
core/summarizer.py
------------------
OpenAI-powered abstractive summariser.

Cost strategy:
  - Compress the lecture text locally first (strip short sentences,
    deduplicate, cap at ~2500 tokens) before sending to the API.
  - One single API call using gpt-4o-mini.
  - Prompt instructs the model to return exactly N bullet sentences
    so we get a clean list back without any extra parsing cost.

Typical cost per summary: ~$0.0005 - $0.001 (half a tenth of a cent).
"""

from __future__ import annotations

import re
from typing import List, Tuple, Union

from core.text_processor import clean_text, split_sentences, LectureTextProcessor
from utils.openai_client import chat, count_tokens

# ── tuneable constants ──────────────────────────────────────────────────────
MAX_INPUT_TOKENS = 2500   # max lecture tokens we send to the API
MIN_SENT_WORDS   = 5      # ignore very short sentences before sending


def _compress_sentences(sentences: List[str], max_tokens: int = MAX_INPUT_TOKENS) -> str:
    """
    Deduplicate and token-cap a sentence list before sending to API.
    Preserves document order.
    """
    seen: set = set()
    kept: List[str] = []
    total = 0
    for s in sentences:
        norm = re.sub(r"\s+", " ", s.strip().lower())
        if norm in seen or len(s.split()) < MIN_SENT_WORDS:
            continue
        seen.add(norm)
        t = count_tokens(s)
        if total + t > max_tokens:
            break
        kept.append(s.strip())
        total += t
    return " ".join(kept)


class TextRankSummarizer:
    """
    Abstractive summariser backed by gpt-4o-mini.
    The name is kept for app.py compatibility.
    """

    def summarize(
        self,
        raw_text_or_sentences: Union[str, List[str]],
        num_sentences: int = 5,
        **_,
    ) -> List[str]:
        """
        Summarise lecture content.

        Parameters
        ----------
        raw_text_or_sentences : str  (raw text)  OR  List[str]  (pre-split sentences)
        num_sentences         : number of bullet points to generate

        Returns
        -------
        List[str]  — one sentence per item, ready for app.py to enumerate.
        """
        # Normalise input
        if isinstance(raw_text_or_sentences, list):
            sentences = [s for s in raw_text_or_sentences if len(s.split()) >= MIN_SENT_WORDS]
        else:
            cleaned, _ = clean_text(raw_text_or_sentences)
            sentences  = split_sentences(cleaned)

        if not sentences:
            return ["No content found to summarise."]

        compressed = _compress_sentences(sentences)
        if not compressed.strip():
            return ["No content found to summarise."]

        system = (
            "You are an expert academic summariser. "
            "Read the lecture text and produce a clear, accurate summary. "
            f"Return EXACTLY {num_sentences} sentences as a numbered list "
            "(1. ... 2. ... etc.). "
            "Cover the most important concepts. No preamble, no extra text."
        )
        user = f"LECTURE TEXT:\n{compressed}"

        raw = chat(system=system, user=user, max_tokens=num_sentences * 60)

        # Parse numbered list robustly
        lines = raw.strip().splitlines()
        results: List[str] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # strip leading "1." / "1)" / "•" / "-"
            cleaned_line = re.sub(r"^[\d]+[.)]\s*|^[-•]\s*", "", line).strip()
            if cleaned_line:
                results.append(cleaned_line)

        return results if results else [raw.strip()]
