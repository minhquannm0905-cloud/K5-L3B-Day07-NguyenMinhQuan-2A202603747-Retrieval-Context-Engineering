import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from raglab.agent import run_agent
from raglab.evaluation import evaluate_retriever
from raglab.providers import OpenAICompatibleClient
from raglab.retrieval import DenseRetriever, HybridRetriever, TfidfRetriever


CHUNKS = [
    {"id": "refund", "text": "Hoàn tiền được xử lý trong 7 ngày làm việc.",
     "source_id": "policy", "source": "policy.pdf", "page": 1,
     "metadata": {"year": 2026}},
    {"id": "ship", "text": "Giao hàng được xử lý trong 3 ngày.",
     "source_id": "policy", "source": "policy.pdf", "page": 2,
     "metadata": {"year": 2026}},
]


class StaticController:
    def decide(self, question, history):
        if not history:
            return {"action": "search", "query": question}
        return {"action": "answer", "answer": "7 ngày làm việc [refund]",
                "citations": ["refund"]}


class SolutionTests(unittest.TestCase):
    def setUp(self):
        self.lexical = TfidfRetriever(CHUNKS)
        vectors = {"refund": [1.0, 0.0], "ship": [0.0, 1.0]}
        self.dense = DenseRetriever(CHUNKS, vectors, lambda query: [1.0, 0.0])

    def test_three_retrieval_strategies(self):
        self.assertEqual(self.lexical.search("hoàn tiền", top_k=1)[0]["id"], "refund")
        self.assertEqual(self.dense.search("refund", top_k=1)[0]["id"], "refund")
        hybrid = HybridRetriever(self.dense, self.lexical)
        self.assertEqual(hybrid.search("hoàn tiền", top_k=1)[0]["id"], "refund")

    def test_filter_and_dimension_validation(self):
        self.assertEqual(self.lexical.search("hoàn tiền", filters={"year": 2025}), [])
        broken = DenseRetriever(CHUNKS, {"refund": [1, 0], "ship": [0, 1]}, lambda q: [1])
        with self.assertRaises(ValueError):
            broken.search("q")

    def test_agent_and_retrieval_metrics(self):
        result = run_agent("Hoàn tiền mất bao lâu?", StaticController(), self.lexical)
        self.assertEqual(result["status"], "answered")
        question = {"id": "q1", "question": "Hoàn tiền mất bao lâu?", "gold": [
            {"source_id": "policy", "page": 1, "evidence": "7 ngày làm việc"}
        ]}
        summary, traces = evaluate_retriever(self.lexical, [question])
        self.assertEqual(summary["hit1"], {"n": 1, "N": 1})
        self.assertTrue(traces[0]["context_hit"])

    def test_local_provider_does_not_require_secret(self):
        client = OpenAICompatibleClient(base_url="http://localhost:11434/v1", api_key="",
                                        chat_model="local-chat", embedding_model="local-embed")
        self.assertEqual(client.api_key, "")
        with self.assertRaises(ValueError):
            OpenAICompatibleClient(base_url="https://example.com/v1", api_key="")


if __name__ == "__main__":
    unittest.main()
