"""
core/mcq_generator.py
---------------------
OpenAI-powered MCQ generator.

Cost strategy:
  - Extract only the top ~1500 tokens of cleaned lecture content locally.
  - Ask gpt-4o-mini to produce ALL questions in ONE single API call
    (not one call per question).
  - Prompt uses strict JSON output so parsing is reliable with zero retries.

Typical cost for 10 MCQs: ~$0.001 (one tenth of a cent).
"""

from __future__ import annotations

import json
import re
from typing import List, Tuple, Union

from core.text_processor import clean_text, split_sentences, LectureTextProcessor
from utils.openai_client import chat, count_tokens

# ── tuneable constants ──────────────────────────────────────────────────────
MAX_INPUT_TOKENS = 1800
MIN_SENT_WORDS   = 5


def _compress(sentences: List[str], max_tokens: int = MAX_INPUT_TOKENS) -> str:
    seen:  set       = set()
    kept:  List[str] = []
    total: int       = 0
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


class MCQGenerator:
    """
    Generates multiple-choice questions from lecture text using gpt-4o-mini.
    """

    def generate(self, raw_text: str, num_questions: int = 5) -> list:
        """Internal method — used by generate_mcqs."""
        return self.generate_mcqs([], num_questions, _raw=raw_text)

    def generate_mcqs(
        self,
        sentences_with_scores: Union[List[Tuple], List[str]],
        num_questions: int = 5,
        _raw: str = "",
    ) -> List[dict]:
        """
        Generate MCQs from lecture content.

        Parameters
        ----------
        sentences_with_scores : List[Tuple[str,float]] from LectureTextProcessor,
                                OR List[str], OR empty list when _raw is provided.
        num_questions         : how many MCQs to generate.

        Returns
        -------
        List[dict] each with keys:
            question      : str
            options       : List[str]  (4 options, A-D)
            correct_index : int        (0-based)
            answer        : str        (correct option text)
        """
        # ── build sentence list ───────────────────────────────────────────
        if _raw:
            cleaned, _ = clean_text(_raw)
            sentences  = split_sentences(cleaned)
        elif sentences_with_scores and isinstance(sentences_with_scores[0], (list, tuple)):
            sentences = [s for s, _ in sentences_with_scores]
        else:
            sentences = list(sentences_with_scores)

        if not sentences:
            return []

        context = _compress(sentences)
        if not context.strip():
            return []

        # ── single API call ───────────────────────────────────────────────
        system = (
            "You are an expert educator creating a multiple-choice quiz. "
            "Read the lecture text carefully and create high-quality MCQs. "
            "Rules:\n"
            "- Each question must be answerable from the lecture text only.\n"
            "- Provide exactly 4 options (A, B, C, D).\n"
            "- Only ONE option is correct.\n"
            "- Distractors must be plausible but clearly wrong on reflection.\n"
            "- Vary question types: definitions, applications, comparisons.\n"
            f"Return ONLY a JSON array of {num_questions} objects. "
            "Each object must have these exact keys:\n"
            '  "question": string\n'
            '  "options": array of 4 strings (do NOT include A/B/C/D prefixes)\n'
            '  "correct_index": integer 0-3\n'
            "No extra text, no markdown, no explanation. Pure JSON only."
        )
        user = f"LECTURE TEXT:\n{context}"

        raw = chat(system=system, user=user, max_tokens=num_questions * 120)

        # ── parse JSON robustly ───────────────────────────────────────────
        # Strip any accidental markdown fences
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

        try:
            items = json.loads(raw)
        except json.JSONDecodeError:
            # Try to extract just the array portion
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if not match:
                return []
            try:
                items = json.loads(match.group())
            except json.JSONDecodeError:
                return []

        results: List[dict] = []
        for item in items:
            try:
                opts    = item["options"]
                c_idx   = int(item["correct_index"])
                if not isinstance(opts, list) or len(opts) != 4:
                    continue
                if not 0 <= c_idx <= 3:
                    continue
                results.append({
                    "question"      : item["question"],
                    "options"       : opts,
                    "correct_index" : c_idx,
                    "answer"        : opts[c_idx],
                })
            except (KeyError, TypeError, ValueError):
                continue

        return results
