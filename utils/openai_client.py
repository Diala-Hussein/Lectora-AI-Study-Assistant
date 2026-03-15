"""
utils/openai_client.py
----------------------
Groq API client — free, no credit card, works instantly.
Uses llama-3.1-8b-instant — fast and capable for Q&A, summaries, MCQs.

Free tier: 14,400 requests/day, 500,000 tokens/day — more than enough.
Get your free key at: console.groq.com
"""

from __future__ import annotations
import os, re
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

MODEL       = "llama-3.1-8b-instant"
TEMPERATURE = 0.3


def _get_client():
    try:
        from groq import Groq
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            raise ValueError(
                "Groq API key not found. "
                "Set GROQ_API_KEY in your .env file or paste it in the sidebar."
            )
        return Groq(api_key=api_key)
    except ImportError:
        raise ImportError("Run: pip install groq")


def chat(
    system: str,
    user: str,
    max_tokens: int = 500,
    temperature: float = TEMPERATURE,
) -> str:
    """Send a prompt to Groq and return the response text."""
    client = _get_client()
    response = client.chat.completions.create(
        model=MODEL,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    )
    return response.choices[0].message.content.strip()


def count_tokens(text: str) -> int:
    """Approximate token count — 1 token ≈ 4 characters."""
    return max(1, len(text) // 4)


def validate_api_key(key: str) -> bool:
    """Groq keys start with 'gsk_'."""
    return bool(re.match(r"^gsk_[A-Za-z0-9]{20,}$", key.strip()))


def set_api_key(key: str) -> None:
    """Override the API key at runtime (Streamlit sidebar)."""
    os.environ["GROQ_API_KEY"] = key.strip()
