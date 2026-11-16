from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from raglab.agent import ModelController, run_agent
from raglab.dense_pgvector import PgvectorDenseRetriever
from raglab.evaluation import evaluate_retriever
from raglab.io_utils import load_env_file, read_json, validate_chunks, write_json
from raglab.providers import OpenAICompatibleClient
from raglab.retrieval import DenseRetriever, HybridRetriever, TfidfRetriever


def load_vectors(path: Path) -> dict[str, list[float]]:
    raw = read_json(path)
    if isinstance(raw, dict) and "vectors" in raw:
        raw = raw["vectors"]
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, list):
        return {row["id"]: row["embedding"] for row in raw}
    raise ValueError("vectors must be an object or an array of id/embedding objects")


def make_client() -> OpenAICompatibleClient:
    return OpenAICompatibleClient()


def build_retriever(args: argparse.Namespace, chunks):
    lexical = TfidfRetriever(chunks)

    if args.strategy == "tfidf":
        return lexical

    database_url = os.getenv("DATABASE_URL", "")
    if not database_url:
        raise ValueError("DATABASE_URL is missing from .env")

    dense = PgvectorDenseRetriever(database_url)

    if args.strategy == "dense":
        return dense

    return HybridRetriever(
        dense,
        lexical,
        candidate_k=args.candidate_k,
        rrf_k=args.rrf_k,
    )


def command_index(args: argparse.Namespace) -> None:
    chunks = validate_chunks(read_json(args.chunks))
    client = make_client()
    vectors = []
    for start in range(0, len(chunks), args.batch_size):
        batch = chunks[start:start + args.batch_size]
        embeddings = client.embed([chunk["text"] for chunk in batch])
        vectors.extend({"id": chunk["id"], "embedding": vector}
                       for chunk, vector in zip(batch, embeddings))
        print(f"embedded {len(vectors)}/{len(chunks)} chunks")
    dimension = len(vectors[0]["embedding"]) if vectors else 0
    write_json(args.output, {"embedding_model": client.embedding_model,
                             "dimension": dimension, "vectors": vectors})


def command_benchmark(args: argparse.Namespace) -> None:
    chunks = validate_chunks(read_json(args.chunks))
    questions = [q for q in read_json(args.questions) if q.get("split") == args.split]
    if not questions:
        raise ValueError(f"no questions found for split {args.split}")
    retriever = build_retriever(args, chunks)
    summary, traces = evaluate_retriever(retriever, questions,
                                         top_k=args.top_k, budget=args.budget)
    summary["split"] = args.split
    summary["embedding_model"] = (
        None if args.strategy == "tfidf" else "intfloat/e5-small-v2"
    )

    write_json(args.output / "summary.json", summary)
    write_json(args.output / "traces.json", traces)
    print(summary)


def command_agent(args: argparse.Namespace) -> None:
    chunks = validate_chunks(read_json(args.chunks))
    cases = [case for case in read_json(args.cases) if case.get("split") == args.split]
    if not cases:
        raise ValueError(f"no cases found for split {args.split}")
    client = make_client()
    retriever = build_retriever(args, chunks)
    traces = []
    for number, case in enumerate(cases, start=1):
        controller = ModelController(client)
        started = time.perf_counter()
        try:
            result = run_agent(case["question"], controller, retriever,
                               filters=case.get("filter"), max_searches=args.max_searches,
                               top_k=args.top_k, budget=args.budget)
            result.update({"query_id": case["id"],
                           "elapsed_ms": (time.perf_counter() - started) * 1000,
                           "model_usage": controller.usage})
        except Exception as exc:
            result = {"query_id": case["id"], "status": "error", "answer": "",
                      "citations": [], "events": [], "error_type": type(exc).__name__,
                      "error": str(exc), "elapsed_ms": (time.perf_counter() - started) * 1000}
        traces.append(result)
        write_json(args.output, traces)
        print(f"{number}/{len(cases)} {case['id']}: {result['status']}")


def command_dense_query(args: argparse.Namespace) -> None:
    database_url = os.getenv("DATABASE_URL", "")
    if not database_url:
        raise ValueError(f"DATABASE_URL is missing from {args.env_file}")
    retriever = PgvectorDenseRetriever(database_url)
    results = retriever.search(args.query, top_k=args.top_k)
    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. {result['id']} | printed_page={result['page']} "
            f"| score={result['score']:.9f}"
        )
    contains = retriever.page_contains_iqr_definition(52)
    print(f"page_52_contains_iqr_definition={'yes' if contains else 'no'}")


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--strategy", choices=("tfidf", "dense", "hybrid"), required=True)
    parser.add_argument("--vectors", type=Path)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--budget", type=int, default=2000)
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument("--rrf-k", type=int, default=60)


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--env-file", type=Path, default=Path(".env"))
    known, _ = pre.parse_known_args()
    load_env_file(known.env_file)

    parser = argparse.ArgumentParser(description="Day 7 Agentic RAG runner", parents=[pre])
    sub = parser.add_subparsers(dest="command", required=True)
    index = sub.add_parser("index")
    index.add_argument("--chunks", type=Path, required=True)
    index.add_argument("--output", type=Path, required=True)
    index.add_argument("--batch-size", type=int, default=32)
    index.set_defaults(func=command_index)

    benchmark = sub.add_parser("benchmark")
    add_common(benchmark)
    benchmark.add_argument("--questions", type=Path, required=True)
    benchmark.add_argument("--split", choices=("dev", "test"), required=True)
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.set_defaults(func=command_benchmark)

    agent = sub.add_parser("agent")
    add_common(agent)
    agent.add_argument("--cases", type=Path, required=True)
    agent.add_argument("--split", choices=("dev", "test"), required=True)
    agent.add_argument("--max-searches", type=int, default=3)
    agent.add_argument("--output", type=Path, required=True)
    agent.set_defaults(func=command_agent)

    dense_query = sub.add_parser("dense-query")
    dense_query.add_argument("--query", required=True)
    dense_query.add_argument("--top-k", type=int, default=3)
    dense_query.set_defaults(func=command_dense_query)

    args = parser.parse_args()
    if hasattr(args, "top_k") and args.top_k < 1:
        parser.error("--top-k must be positive")
    if hasattr(args, "budget") and args.budget < 1:
        parser.error("--budget must be positive")
    args.func(args)


if __name__ == "__main__":
    main()
