from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REQUIRED_CHUNK_FIELDS = ("id", "text", "source_id", "source", "page", "metadata")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def validate_chunks(chunks: Any) -> list[dict[str, Any]]:
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("chunks must be a non-empty JSON array")
    seen: set[str] = set()
    for chunk in chunks:
        if not isinstance(chunk, dict):
            raise ValueError("each chunk must be an object")
        missing = [field for field in REQUIRED_CHUNK_FIELDS if field not in chunk]
        if missing:
            raise ValueError(f"chunk is missing fields: {', '.join(missing)}")
        if not isinstance(chunk["id"], str) or not chunk["id"].strip():
            raise ValueError("chunk id must be a non-empty string")
        if chunk["id"] in seen:
            raise ValueError(f"duplicate chunk id: {chunk['id']}")
        if not isinstance(chunk["text"], str) or not chunk["text"].strip():
            raise ValueError(f"chunk {chunk['id']} has empty text")
        if not isinstance(chunk["metadata"], dict):
            raise ValueError(f"chunk {chunk['id']} metadata must be an object")
        seen.add(chunk["id"])
    return chunks


def load_env_file(path: str | Path) -> None:
    """Load a small KEY=VALUE file without overwriting process variables."""
    import os

    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
