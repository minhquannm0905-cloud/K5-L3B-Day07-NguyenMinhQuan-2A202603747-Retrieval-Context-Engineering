"""Convert Day 6 JSONL chunks and questions into the Day 7 data contract."""
import argparse
import json
from pathlib import Path


def read_json_or_jsonl(path: Path):
    """Read either a JSON array or a JSON Lines file."""
    raw = path.read_text(encoding="utf-8-sig").strip()

    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    return [
        json.loads(line)
        for line in raw.splitlines()
        if line.strip()
    ]


def convert_chunk(chunk):
    metadata = chunk.get("metadata")
    if metadata is None:
        metadata = {
            "chapter": chunk["chapter"],
            "pdf_page": chunk["pdf_page"],
            "page_id": chunk["page_id"],
            "strategy": chunk["strategy"],
        }

    return {
        "id": chunk["chunk_id"],
        "source_id": chunk["document_id"],
        "source": chunk["source"],
        "page": chunk.get("printed_page", chunk.get("page")),
        "text": chunk["text"],
        "metadata": metadata,
    }


def convert_question(question):
    result = {
        "id": question["id"],
        "split": question["split"],
        "question": question["question"],
        "expected_answer": question.get("answer", question.get("expected_answer")),
        "gold": [
            {
                "source_id": question.get("document_id", question.get("gold_source_id")),
                "page": question.get("printed_page", question.get("gold_page")),
                "evidence": question.get("evidence", question.get("evidence_quote")),
            }
        ],
    }

    if "filters" in question:
        result["filter"] = question["filters"]
    elif "filter" in question:
        result["filter"] = question["filter"]

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", required=True, type=Path)
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    raw_chunks = read_json_or_jsonl(args.chunks)
    raw_questions = read_json_or_jsonl(args.questions)

    chunks = [convert_chunk(chunk) for chunk in raw_chunks]
    questions = [convert_question(question) for question in raw_questions]

    args.output.mkdir(parents=True, exist_ok=True)

    (args.output / "chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output / "questions.json").write_text(
        json.dumps(questions, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Saved {len(chunks)} chunks to {args.output / 'chunks.json'}")
    print(f"Saved {len(questions)} questions to {args.output / 'questions.json'}")


if __name__ == "__main__":
    main()
