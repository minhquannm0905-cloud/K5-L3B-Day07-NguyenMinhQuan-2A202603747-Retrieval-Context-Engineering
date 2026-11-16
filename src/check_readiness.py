from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from raglab.io_utils import read_json, validate_chunks


MINIMUMS = {("dev", True): 15, ("dev", False): 2,
            ("test", True): 5, ("test", False): 2}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Day 7 data before experiments")
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    args = parser.parse_args()

    chunks = validate_chunks(read_json(args.chunks))
    questions = read_json(args.questions)
    cases = read_json(args.cases)
    chunk_ids = {chunk["id"] for chunk in chunks}
    case_ids = {case.get("id") for case in cases}
    errors: list[str] = []
    counts: Counter[tuple[str, bool]] = Counter()

    if len(case_ids) != len(cases) or None in case_ids:
        errors.append("agent case IDs must be present and unique")
    for question in questions:
        split = question.get("split")
        answerable = question.get("answerable") is not False
        counts[(split, answerable)] += 1
        if answerable:
            if not question.get("gold") or not question.get("expected_answer"):
                errors.append(f"{question.get('id')}: answerable question lacks gold/expected_answer")
            for gold in question.get("gold", []):
                resolved = any(chunk["source_id"] == gold.get("source_id")
                               and chunk["page"] == gold.get("page")
                               and gold.get("evidence", "") in chunk["text"] for chunk in chunks)
                if not resolved:
                    errors.append(f"{question.get('id')}: gold evidence does not resolve to a chunk")
        if question.get("id") not in case_ids:
            errors.append(f"{question.get('id')}: missing matching agent case")

    for key, minimum in MINIMUMS.items():
        if counts[key] < minimum:
            errors.append(f"{key[0]} {'answerable' if key[1] else 'unanswerable'}: "
                          f"need {minimum}, found {counts[key]}")

    print(f"chunks: {len(chunks)}")
    for key in MINIMUMS:
        print(f"{key[0]} {'answerable' if key[1] else 'unanswerable'}: {counts[key]}")
    if errors:
        print("\nNOT READY")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("\nREADY: data meets the minimum Day 7 benchmark contract")


if __name__ == "__main__":
    main()
