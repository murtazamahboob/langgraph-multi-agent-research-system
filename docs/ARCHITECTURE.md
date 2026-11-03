# Architecture

State is a single `ResearchState` dict-like object passed node to node,
exactly like a LangGraph state graph:

- `planner_node` fills `subquestions`
- the research loop fills `evidence` and per-question hits
- `critic_node` sets `coverage` and `verdict`
- `synthesizer_node` fills `brief`

The conditional edge is in `run_research`: loop while verdict is
"revise" and rounds < MAX_ROUNDS (2). This is the termination guarantee.

To go production: replace `retrieve` in `src/retrieval.py` with a vector
store call, and replace the keyword planner with an LLM planner that
returns the same list-of-questions shape. No node signatures change.
