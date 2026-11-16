import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from raglab.dense_pgvector import (
    DOCUMENT_ID,
    EMBEDDING_DIMENSION,
    INDEXED_STRATEGY,
    E5QueryEmbedder,
    PgvectorDenseRetriever,
)


class FakeVector:
    def tolist(self):
        return [1.0] + [0.0] * (EMBEDDING_DIMENSION - 1)


class FakeModel:
    def __init__(self):
        self.calls = []

    def encode(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return FakeVector()


class FakeCursor:
    def __init__(self, connection):
        self.connection = connection
        self.sql = None
        self.parameters = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, sql, parameters):
        self.sql = sql
        self.parameters = parameters
        self.connection.calls.append((sql, parameters))

    def fetchall(self):
        return [{"chunk_id": "b-p52-001", "content": "IQR definition",
                 "document_id": DOCUMENT_ID, "source": "book.pdf",
                 "printed_page": 52, "chapter": 4, "score": 0.91}]

    def fetchone(self):
        return {"contains_definition": True}


class FakeConnection:
    def __init__(self):
        self.read_only = False
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def cursor(self, **kwargs):
        return FakeCursor(self)


class DensePgvectorTests(unittest.TestCase):
    def test_e5_query_prefix_and_normalization(self):
        model = FakeModel()
        vector = E5QueryEmbedder(model)("How is IQR calculated?")
        self.assertEqual(len(vector), EMBEDDING_DIMENSION)
        self.assertEqual(model.calls[0][0], "query: How is IQR calculated?")
        self.assertIs(model.calls[0][1]["normalize_embeddings"], True)

    def test_read_only_parameterized_strategy_b_mapping(self):
        connections = []

        def connect(database_url):
            connection = FakeConnection()
            connections.append(connection)
            return connection

        retriever = PgvectorDenseRetriever(
            "postgresql://not-printed",
            embed_query=lambda query: [1.0] + [0.0] * (EMBEDDING_DIMENSION - 1),
            connect=connect,
        )
        rows = retriever.search("IQR", top_k=3)
        self.assertTrue(connections[0].read_only)
        parameters = connections[0].calls[0][1]
        self.assertEqual(parameters["strategy"], INDEXED_STRATEGY)
        self.assertEqual(parameters["document_id"], DOCUMENT_ID)
        self.assertEqual(parameters["top_k"], 3)
        self.assertEqual(rows[0]["id"], "b-p52-001")
        self.assertEqual(rows[0]["text"], "IQR definition")
        self.assertEqual(rows[0]["source_id"], DOCUMENT_ID)
        self.assertEqual(rows[0]["page"], 52)
        self.assertEqual(rows[0]["metadata"], {"chapter": 4})

    def test_page_definition_check_is_read_only(self):
        connections = []

        def connect(database_url):
            connection = FakeConnection()
            connections.append(connection)
            return connection

        retriever = PgvectorDenseRetriever(
            "postgresql://not-printed",
            embed_query=lambda query: [0.0] * EMBEDDING_DIMENSION,
            connect=connect,
        )
        self.assertTrue(retriever.page_contains_iqr_definition(52))
        self.assertTrue(connections[0].read_only)
        self.assertEqual(connections[0].calls[0][1]["printed_page"], 52)


if __name__ == "__main__":
    unittest.main()
