"""Planner agent: turn one research question into sub-questions.

Rules, kept deliberately simple and deterministic:
- A query joined by " and " or commas is split into its parts.
- Otherwise the core topic is expanded into three angles: definition,
  mechanism, and practical use / limits. That gives the researcher
  distinct angles to retrieve against instead of one vague query.
"""

import re

from .retrieval import STOPWORDS


def core_topic(query: str) -> str:
    words = re.findall(r"[a-z0-9]+", query.lower())
    kept = [w for w in words if w not in STOPWORDS]
    return " ".join(kept) if kept else query.strip()


def plan(query: str) -> list:
    if not query or not query.strip():
        raise ValueError("query must not be empty")
    cleaned = query.strip().rstrip("?")
    parts = [p.strip() for p in re.split(r"\s+and\s+|,\s*", cleaned) if p.strip()]
    if len(parts) > 1:
        return [f"{p}?" if not p.endswith("?") else p for p in parts]
    topic = core_topic(cleaned)
    return [
        f"What is {topic}?",
        f"How does {topic} work in practice?",
        f"What are the practical uses and limits of {topic}?",
    ]
