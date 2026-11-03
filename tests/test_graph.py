import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.graph import ResearchState, critic_node, researcher_node, run_research, synthesizer_node
from src.planner import plan
from src.report import render_markdown
from src.retrieval import load_corpus, retrieve

CORPUS = load_corpus()


class TestPlanner(unittest.TestCase):
    def test_multi_part_query_splits(self):
        self.assertEqual(len(plan("LangGraph and RAG")), 2)

    def test_single_topic_gets_three_angles(self):
        self.assertEqual(len(plan("How does LangGraph work?")), 3)

    def test_empty_query_raises(self):
        with self.assertRaises(ValueError):
            plan("   ")


class TestRetrieval(unittest.TestCase):
    def test_opencv_query_ranks_opencv_doc_first(self):
        results = retrieve("OpenCV object detection real time", CORPUS)
        self.assertTrue(results)
        self.assertEqual(results[0]["id"], "doc-opencv")

    def test_no_match_returns_nothing_not_fake_evidence(self):
        self.assertEqual(retrieve("xyzzy blorp", CORPUS), [])


class TestGraph(unittest.TestCase):
    def test_researcher_dedupes_same_doc(self):
        state = ResearchState(query="test")
        state.subquestions = ["LangGraph state graphs", "LangGraph multi-agent systems"]
        researcher_node(state, CORPUS)
        ids = [d["id"] for d in state.evidence]
        self.assertEqual(len(ids), len(set(ids)))

    def test_critic_requests_revision_when_a_question_is_uncovered(self):
        state = ResearchState(query="test")
        state.subquestions = ["covered?", "uncovered?"]
        state._question_hits = {"covered?": [{"id": "doc-langgraph"}], "uncovered?": []}
        critic_node(state)
        self.assertEqual(state.verdict, "revise")
        self.assertEqual(state.coverage, 0.5)

    def test_full_run_terminates_with_citations(self):
        brief = run_research("How does LangGraph multi-agent research work?", CORPUS)
        self.assertLessEqual(brief["rounds"], 2)
        self.assertTrue(brief["citations"])
        self.assertTrue(brief["findings"])

    def test_synthesizer_lists_gaps_honestly(self):
        state = ResearchState(query="test", coverage=0.5)
        state.subquestions = ["covered?", "uncovered?"]
        state._question_hits = {
            "covered?": [{"id": "doc-langgraph", "title": "LangGraph", "text": "graph text"}],
            "uncovered?": [],
        }
        brief = synthesizer_node(state).brief
        self.assertEqual(brief["gaps"], ["uncovered?"])

    def test_empty_query_raises(self):
        with self.assertRaises(ValueError):
            run_research("   ", CORPUS)

    def test_markdown_states_gaps_for_unknown_topic(self):
        brief = run_research("xyzzy blorp quantum", CORPUS)
        rendered = render_markdown(brief)
        self.assertIn("Gaps", rendered)
        self.assertIn("None", rendered)


if __name__ == "__main__":
    unittest.main()
