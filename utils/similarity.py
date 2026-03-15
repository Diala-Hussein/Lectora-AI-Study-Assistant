"""
utils/similarity.py
-------------------
Vectorized similarity utilities using numpy.
Provides cosine similarity, BM25-style keyword overlap,
and a hybrid scorer used across QA, summarization, and MCQ modules.
"""

from __future__ import annotations

import re
from typing import List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

def cosine_similarity_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    Compute an (N x N) cosine-similarity matrix for a row-normalised
    embedding matrix.

    Parameters
    ----------
    matrix : np.ndarray, shape (N, D)
        Each row is an embedding vector (need not be unit-normalised).

    Returns
    -------
    np.ndarray, shape (N, N)
        Symmetric matrix where entry [i, j] is cos_sim(row_i, row_j).
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    normed = matrix / norms
    return normed @ normed.T


def cosine_similarity_1d(vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Cosine similarity between a single vector and every row of a matrix.

    Parameters
    ----------
    vec    : np.ndarray, shape (D,)
    matrix : np.ndarray, shape (N, D)

    Returns
    -------
    np.ndarray, shape (N,) – similarity scores.
    """
    vec_norm = vec / (np.linalg.norm(vec) + 1e-10)
    mat_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    mat_norms = np.where(mat_norms == 0, 1e-10, mat_norms)
    normed = matrix / mat_norms
    return normed @ vec_norm


# ---------------------------------------------------------------------------
# Keyword / token overlap (BM25-lite)
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would could should may might shall can this that these those "
    "i me my we our you your he she it its they them their what which "
    "who whom when where why how all each every both few more most other "
    "some such no nor not only own same so than too very just but and or "
    "in on at to for of with by about against between into through during "
    "before after above below from up down out off over under again further "
    "then once there here".split()
)


def tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, remove stopwords."""
    tokens = re.findall(r"\b[a-z]{2,}\b", text.lower())
    return [t for t in tokens if t not in _STOPWORDS]


def keyword_overlap_score(query_tokens: List[str], candidate_tokens: List[str]) -> float:
    """
    Jaccard-style keyword overlap score between two token lists.

    Returns a float in [0, 1].
    """
    if not query_tokens or not candidate_tokens:
        return 0.0
    q_set = set(query_tokens)
    c_set = set(candidate_tokens)
    intersection = q_set & c_set
    union = q_set | c_set
    return len(intersection) / len(union)


def batch_keyword_overlap(query_tokens: List[str], candidates: List[List[str]]) -> np.ndarray:
    """
    Vectorised keyword overlap for a list of candidate token lists.

    Returns np.ndarray of shape (N,).
    """
    return np.array([keyword_overlap_score(query_tokens, c) for c in candidates])


# ---------------------------------------------------------------------------
# Sentence importance heuristic
# ---------------------------------------------------------------------------

def sentence_importance_scores(sentences: List[str]) -> np.ndarray:
    """
    Heuristic importance score for each sentence based on:
      - token count (longer → more informative, up to a ceiling)
      - presence of numeric data
      - position bonus for early sentences

    Returns np.ndarray of shape (N,), values in [0, 1].
    """
    scores = []
    n = len(sentences)
    for idx, sent in enumerate(sentences):
        tokens = tokenize(sent)
        length_score = min(len(tokens) / 20.0, 1.0)
        numeric_score = 0.3 if re.search(r"\d", sent) else 0.0
        position_score = 0.2 * max(0, 1 - idx / max(n, 1))
        scores.append(length_score + numeric_score + position_score)

    scores_arr = np.array(scores, dtype=float)
    max_s = scores_arr.max()
    if max_s > 0:
        scores_arr /= max_s
    return scores_arr


# ---------------------------------------------------------------------------
# Hybrid scorer
# ---------------------------------------------------------------------------

def hybrid_score(
    semantic_scores: np.ndarray,
    keyword_scores: np.ndarray,
    importance_scores: np.ndarray,
    alpha: float = 0.6,
    beta: float = 0.25,
    gamma: float = 0.15,
) -> np.ndarray:
    """
    Combine three score arrays into a single hybrid ranking vector.

    Parameters
    ----------
    semantic_scores   : cosine-similarity scores, shape (N,)
    keyword_scores    : keyword-overlap scores,   shape (N,)
    importance_scores : sentence-importance,      shape (N,)
    alpha, beta, gamma: weights (should sum to ~1).

    Returns
    -------
    np.ndarray, shape (N,)
    """
    return alpha * semantic_scores + beta * keyword_scores + gamma * importance_scores
