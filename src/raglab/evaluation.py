from __future__ import annotations
import re
import unicodedata
import statistics
import time
from typing import Any

from .agent import pack_context

def evidence_key(text: str) -> str:
    """Normalize only for robust evidence matching."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"(?<=\w)-[ \t]*\r?\n[ \t]*(?=\w)", "", text)
    text = text.replace("\u00ad", "").replace("\u2212", "-")
    text = text.translate(str.maketrans("\u201c\u201d", '""'))
    return "".join(text.split())


def evidence_hit(chunk: dict[str, Any], gold: list[dict[str, Any]]) -> bool:
    chunk_key = evidence_key(chunk["text"])

    return any(
        chunk["source_id"] == item["source_id"]
        and chunk["page"] == item["page"]
        and evidence_key(item["evidence"]) in chunk_key
        for item in gold
    )

def evaluate_retriever(retriever: Any, questions: list[dict[str, Any]], *,
                       top_k: int = 3, budget: int = 2000) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    traces = []
    for question in questions:
        if question.get("answerable") is False:
            continue
        started = time.perf_counter()
        results = retriever.search(question["question"], filters=question.get("filter"),
                                   top_k=max(3, top_k))
        elapsed_ms = (time.perf_counter() - started) * 1000
        marked = [{**row, "hit": evidence_hit(row, question["gold"])} for row in results]
        context, ids = pack_context(marked[:top_k], budget)
        traces.append({"query_id": question["id"], "question": question["question"],
                       "strategy": getattr(retriever, "name", type(retriever).__name__),
                       "top_k": top_k, "budget_chars": budget, "results": marked,
                       "context": context, "context_ids": ids, "elapsed_ms": elapsed_ms,
                       "hit1": bool(marked and marked[0]["hit"]),
                       "hit3": any(row["hit"] for row in marked[:3]),
                       "context_hit": any(row["hit"] and row["id"] in ids for row in marked)})
    count = len(traces)
    summary = {"strategy": getattr(retriever, "name", type(retriever).__name__), "n": count,
               "hit1": {"n": sum(row["hit1"] for row in traces), "N": count},
               "evidence_hit3": {"n": sum(row["hit3"] for row in traces), "N": count},
               "context_evidence_rate": {"n": sum(row["context_hit"] for row in traces), "N": count},
               "median_ms": statistics.median(row["elapsed_ms"] for row in traces) if traces else None,
               "top_k": top_k, "budget_chars": budget}
    return summary, traces
