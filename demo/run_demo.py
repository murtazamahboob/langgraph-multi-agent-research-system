"""Demo: run the agent graph on sample queries, no API key needed."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.graph import run_research
from src.report import render_markdown

base = os.path.join(os.path.dirname(__file__), "..", "samples", "queries.json")
with open(base) as f:
    queries = json.load(f)

for query in queries:
    print(render_markdown(run_research(query)))
    print("=" * 60)
