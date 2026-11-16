from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Any, Callable


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", unicodedata.normalize("NFC", text).casefold())


def metadata_matches(chunk: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    if not filters:
        return True
    metadata = chunk.get("metadata", {})
    return all(chunk.get(key, metadata.get(key)) == expected for key, expected in filters.items())


class TfidfRetriever:
    name = "tfidf"

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self.chunks = chunks
        counts = [Counter(tokenize(chunk["text"])) for chunk in chunks]
        df = Counter(term for row in counts for term in row)
        self.idf = {term: math.log((1 + len(chunks)) / (1 + count)) + 1 for term, count in df.items()}
        self.vectors = [self._vector(row) for row in counts]

    def _vector(self, counts: Counter[str]) -> dict[str, float]:
        values = {term: count * self.idf[term] for term, count in counts.items() if term in self.idf}
        norm = math.sqrt(sum(value * value for value in values.values())) or 1
        return {term: value / norm for term, value in values.items()}

    def search(self, query: str, *, filters: dict[str, Any] | None = None,
               top_k: int = 3) -> list[dict[str, Any]]:
        query_vector = self._vector(Counter(tokenize(query)))
        results = []
        for chunk, vector in zip(self.chunks, self.vectors):
            if not metadata_matches(chunk, filters):
                continue
            score = sum(value * vector.get(term, 0) for term, value in query_vector.items())
            if score > 0:
                results.append({**chunk, "score": score, "strategy": self.name})
        return sorted(results, key=lambda item: (-item["score"], item["id"]))[:top_k]


class DenseRetriever:
    name = "dense"

    def __init__(self, chunks: list[dict[str, Any]], vectors: dict[str, list[float]],
                 embed_query: Callable[[str], list[float]]) -> None:
        self.chunks = chunks
        self.vectors = vectors
        self.embed_query = embed_query
        missing = [chunk["id"] for chunk in chunks if chunk["id"] not in vectors]
        if missing:
            raise ValueError(f"missing vectors for {len(missing)} chunks; first: {missing[0]}")
        dimensions = {len(vectors[chunk["id"]]) for chunk in chunks}
        if len(dimensions) != 1 or 0 in dimensions:
            raise ValueError("document vectors must have one non-zero dimension")
        self.dimension = next(iter(dimensions))

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        if len(left) != len(right):
            raise ValueError("query and document embedding dimensions do not match")
        denominator = math.sqrt(sum(x * x for x in left)) * math.sqrt(sum(x * x for x in right))
        return sum(x * y for x, y in zip(left, right)) / denominator if denominator else 0

    def search(self, query: str, *, filters: dict[str, Any] | None = None,
               top_k: int = 3) -> list[dict[str, Any]]:
        query_vector = self.embed_query(query)
        if len(query_vector) != self.dimension:
            raise ValueError(f"query embedding has {len(query_vector)} dimensions; expected {self.dimension}")
        results = []
        for chunk in self.chunks:
            if metadata_matches(chunk, filters):
                score = self._cosine(query_vector, self.vectors[chunk["id"]])
                results.append({**chunk, "score": score, "strategy": self.name})
        return sorted(results, key=lambda item: (-item["score"], item["id"]))[:top_k]


class HybridRetriever:
    name = "hybrid_rrf"

    def __init__(self, dense: DenseRetriever, lexical: TfidfRetriever, *,
                 candidate_k: int = 10, rrf_k: int = 60) -> None:
        self.dense = dense
        self.lexical = lexical
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k

    def search(self, query: str, *, filters: dict[str, Any] | None = None,
               top_k: int = 3) -> list[dict[str, Any]]:
        limit = max(top_k, self.candidate_k)
        rankings = [self.dense.search(query, filters=filters, top_k=limit),
                    self.lexical.search(query, filters=filters, top_k=limit)]
        scores: dict[str, float] = {}
        chunks: dict[str, dict[str, Any]] = {}
        ranks: dict[str, dict[str, int]] = {}
        for ranking in rankings:
            strategy = ranking[0]["strategy"] if ranking else "empty"
            for rank, chunk in enumerate(ranking, start=1):
                chunk_id = chunk["id"]
                scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (self.rrf_k + rank)
                chunks[chunk_id] = chunk
                ranks.setdefault(chunk_id, {})[strategy] = rank
        ordered = sorted(chunks, key=lambda cid: (-scores[cid], cid))[:top_k]
        return [{**chunks[cid], "score": scores[cid], "strategy": self.name,
                 "component_ranks": ranks[cid]} for cid in ordered]
