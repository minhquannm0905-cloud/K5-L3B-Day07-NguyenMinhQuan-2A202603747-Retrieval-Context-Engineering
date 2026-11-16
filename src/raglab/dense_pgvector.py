from __future__ import annotations

import json
import math
from typing import Any, Callable


MODEL_NAME = "intfloat/e5-small-v2"
EMBEDDING_DIMENSION = 384
INDEXED_STRATEGY = "B"
DOCUMENT_ID = "thinkstats2-c3-c5"
IQR_DEFINITION = "The IQR is the difference between the 75th and 25th percentiles."


SEARCH_SQL = """
SELECT chunk_id, content, document_id, source, printed_page, chapter,
       1 - (embedding <=> %(query_vector)s::vector) AS score
FROM lab_chunks
WHERE strategy = %(strategy)s
  AND document_id = %(document_id)s
    AND (%(chapter)s::integer IS NULL OR chapter = %(chapter)s::integer)
ORDER BY embedding <=> %(query_vector)s::vector, chunk_id
LIMIT %(top_k)s
"""


PAGE_CONTAINS_SQL = """
SELECT EXISTS (
    SELECT 1
    FROM lab_chunks
    WHERE strategy = %(strategy)s
      AND document_id = %(document_id)s
      AND printed_page = %(printed_page)s
      AND content ILIKE '%%' || %(needle)s || '%%'
) AS contains_definition
"""


class E5QueryEmbedder:
    """Day 06-compatible local query embedding."""

    def __init__(self, model: Any | None = None) -> None:
        if model is None:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(MODEL_NAME, device="cpu", local_files_only=True)
        self.model = model

    def __call__(self, query: str) -> list[float]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        vector = self.model.encode(
            f"query: {query.strip()}",
            normalize_embeddings=True,
        )
        values = vector.tolist() if hasattr(vector, "tolist") else list(vector)
        if len(values) != EMBEDDING_DIMENSION:
            raise ValueError(
                f"query embedding has {len(values)} dimensions; expected {EMBEDDING_DIMENSION}"
            )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("query embedding contains a non-finite value")
        return [float(value) for value in values]


class PgvectorDenseRetriever:
    """Read-only dense retrieval over Day 06 Strategy B chunks."""

    name = "dense_pgvector_strategy_b"

    def __init__(
        self,
        database_url: str,
        *,
        embed_query: Callable[[str], list[float]] | None = None,
        connect: Callable[..., Any] | None = None,
    ) -> None:
        if not database_url:
            raise ValueError("DATABASE_URL is required")
        if connect is None:
            import psycopg
            from psycopg.rows import dict_row

            connect = psycopg.connect
            self.row_factory = dict_row
        else:
            self.row_factory = None
        self.database_url = database_url
        self.embed_query = embed_query or E5QueryEmbedder()
        self.connect = connect

    @staticmethod
    def _vector_parameter(vector: list[float]) -> str:
        if len(vector) != EMBEDDING_DIMENSION:
            raise ValueError(
                f"query embedding has {len(vector)} dimensions; expected {EMBEDDING_DIMENSION}"
            )
        return json.dumps(vector, allow_nan=False, separators=(",", ":"))

    @staticmethod
    def _set_read_only(connection: Any) -> None:
        # psycopg applies this to the next transaction before any statement runs.
        connection.read_only = True

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        filters = filters or {}
        unknown_filters = set(filters) - {"chapter"}

        if unknown_filters:
            raise ValueError(
                f"Unsupported dense filters: {sorted(unknown_filters)}"
            )

        chapter = filters.get("chapter")

        if chapter is not None and not isinstance(chapter, int):
            raise ValueError("chapter filter must be an integer")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        query_vector = self._vector_parameter(self.embed_query(query))

        parameters = {
            "query_vector": query_vector,
            "strategy": INDEXED_STRATEGY,
            "document_id": DOCUMENT_ID,
            "chapter": chapter,
            "top_k": top_k,
        }
        with self.connect(self.database_url) as connection:
            self._set_read_only(connection)
            with connection.cursor(row_factory=self.row_factory) as cursor:
                cursor.execute(SEARCH_SQL, parameters)
                rows = cursor.fetchall()
        return [
            {
                "id": row["chunk_id"],
                "text": row["content"],
                "source_id": row["document_id"],
                "source": row["source"],
                "page": row["printed_page"],
                "metadata": {"chapter": row["chapter"]},
                "score": float(row["score"]),
                "strategy": self.name,
            }
            for row in rows
        ]

    def page_contains_iqr_definition(self, printed_page: int = 52) -> bool:
        parameters = {
            "strategy": INDEXED_STRATEGY,
            "document_id": DOCUMENT_ID,
            "printed_page": printed_page,
            "needle": IQR_DEFINITION,
        }
        with self.connect(self.database_url) as connection:
            self._set_read_only(connection)
            with connection.cursor(row_factory=self.row_factory) as cursor:
                cursor.execute(PAGE_CONTAINS_SQL, parameters)
                row = cursor.fetchone()
        return bool(row and row["contains_definition"])
