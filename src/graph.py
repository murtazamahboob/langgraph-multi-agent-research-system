"""The agent graph: Planner -> Researcher -> Critic -> (loop) -> Synthesizer.

This mirrors a LangGraph state graph with conditional edges, but the
core runs on the standard library so it is testable without an LLM
key or the langgraph package. Swap `retrieve` for an LLM-backed
retriever and the node/edge structure stays the same.

State flows through the nodes as one dict:
  query, subquestions, evidence (deduped docs), rounds,
  coverage, verdict, brief
"""

from dataclasses import dataclass, field

from .planner import plan
from .retrieval import load_corpus, retrieve

MAX_ROUNDS = 2


@dataclass
class ResearchState:
    query: str
    subquestions: list = field(default_factory=list)
    evidence: list = field(default_factory=list)
    rounds: int = 0
    coverage: float = 0.0
    verdict: str = "pending"
    brief: dict = field(default_factory=dict)


def planner_node(state: ResearchState) -> ResearchState:
    state.subquestions = plan(state.query)
    return state


def researcher_node(state: ResearchState, corpus: list, top_k: int = 3) -> ResearchState:
    """Retrieve for every sub-question and dedupe by document id."""
    seen = {doc["id"] for doc in state.evidence}
    for question in state.subquestions:
        for doc in retrieve(question, corpus, top_k=top_k):
            if doc["id"] not in seen:
                seen.add(doc["id"])
                state.evidence.append(doc)
    state.rounds += 1
    return state


def critic_node(state: ResearchState) -> ResearchState:
    """Coverage = share of sub-questions with at least one retrieved doc.

    A doc counts for a sub-question only if it was retrieved for that
    question, so coverage is recomputed honestly here instead of
    assuming every doc covers every question.
    """
    # Recompute per-question coverage from scratch is not possible from
    # deduped evidence alone, so the researcher records it on the state
    # via _question_hits (see run_research). Fall back to a simple rule:
    hits = getattr(state, "_question_hits", None)
    if hits is not None:
        covered = sum(1 for q in state.subquestions if hits.get(q))
    else:
        covered = len(state.subquestions) if state.evidence else 0
    total = len(state.subquestions) or 1
    state.coverage = covered / total
    state.verdict = "approved" if state.coverage >= 1.0 else "revise"
    return state


def synthesizer_node(state: ResearchState) -> ResearchState:
    findings = []
    gaps = []
    for question in state.subquestions:
        hits = getattr(state, "_question_hits", {}).get(question, [])
        if hits:
            top = hits[0]
            findings.append({
                "question": question,
                "answer": top["text"][:300],
                "source_id": top["id"],
                "source_title": top["title"],
            })
        else:
            gaps.append(question)
    state.brief = {
        "query": state.query,
        "summary": (
            f"Research brief on '{state.query}' covering "
            f"{len(findings)} of {len(state.subquestions)} angles, "
            f"in {state.rounds} research round(s)."
        ),
        "findings": findings,
        "gaps": gaps,
        "citations": sorted({f["source_id"] for f in findings}),
        "confidence": round(state.coverage, 2),
        "rounds": state.rounds,
    }
    return state


def run_research(query: str, corpus: list = None) -> dict:
    """Run the full graph. Always terminates: at most MAX_ROUNDS loops."""
    if not query or not query.strip():
        raise ValueError("query must not be empty")
    corpus = corpus if corpus is not None else load_corpus()
    state = ResearchState(query=query.strip())
    state = planner_node(state)

    # Track per-question hits so the critic and synthesizer stay honest.
    state._question_hits = {}
    while True:
        top_k = 3 if state.rounds == 0 else 5  # wider net on the revision round
        seen = {doc["id"] for doc in state.evidence}
        for question in state.subquestions:
            docs = retrieve(question, corpus, top_k=top_k)
            state._question_hits.setdefault(question, [])
            for doc in docs:
                if doc["id"] not in {d["id"] for d in state._question_hits[question]}:
                    state._question_hits[question].append(doc)
                if doc["id"] not in seen:
                    seen.add(doc["id"])
                    state.evidence.append(doc)
        state.rounds += 1
        state = critic_node(state)
        # Conditional edge: revise only if uncovered AND rounds remain.
        if state.verdict == "approved" or state.rounds >= MAX_ROUNDS:
            break

    state = synthesizer_node(state)
    return state.brief
