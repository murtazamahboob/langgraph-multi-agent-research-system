"""Keyword retrieval over a small local corpus.

Pure standard library, so the whole system runs and tests offline
with no API key. In production the same interface can be backed by
a vector store; the graph code does not care where scores come from.
"""

import json
import os
import re

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "what", "how", "why",
    "when", "where", "which", "who", "do", "does", "did", "can", "could",
    "should", "would", "and", "or", "but", "of", "for", "to", "in", "on",
    "with", "by", "about", "explain", "tell", "me", "please", "i", "we",
    # Planner-template filler: these words appear in every generated
    # angle ("how does X work in practice?"), so counting them would
    # match almost any document and hide real gaps.
    "work", "works", "practice", "practical", "uses", "use", "limits",
    "limit",
}


def tokenize(text: str) -> set:
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 1}


def score_document(query_tokens: set, doc: dict) -> float:
    """Title matches count double; body matches count once."""
    title_tokens = tokenize(doc.get("title", ""))
    body_tokens = tokenize(doc.get("text", ""))
    tag_tokens = {t.lower() for t in doc.get("tags", [])}
    title_hits = len(query_tokens & title_tokens)
    body_hits = len(query_tokens & body_tokens)
    tag_hits = len(query_tokens & tag_tokens)
    return 2.0 * title_hits + 1.0 * body_hits + 1.5 * tag_hits


def retrieve(query: str, corpus: list, top_k: int = 3) -> list:
    """Return the top_k docs for query, each with a score, best first.

    Docs scoring 0 are dropped: no evidence is better than fake evidence,
    and the critic agent relies on that to detect gaps honestly.
    """
    tokens = tokenize(query)
    if not tokens:
        return []
    scored = []
    for doc in corpus:
        score = score_document(tokens, doc)
        if score > 0:
            scored.append({**doc, "score": score})
    scored.sort(key=lambda d: (-d["score"], d["id"]))
    return scored[:top_k]


def load_corpus(path: str = None) -> list:
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "..", "samples", "corpus.json")
    with open(path) as f:
        return json.load(f)
