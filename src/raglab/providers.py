from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


class ProviderError(RuntimeError):
    pass


class OpenAICompatibleClient:
    """Minimal client for OpenAI-compatible embeddings and chat endpoints."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        chat_model: str | None = None,
        embedding_model: str | None = None,
        timeout: float = 250,
        retries: int = 2,
    ) -> None:
        self.base_url = (base_url or os.getenv("RAG_API_BASE", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key if api_key is not None else os.getenv("RAG_API_KEY", "")
        self.chat_model = chat_model or os.getenv("RAG_CHAT_MODEL", "")
        self.embedding_model = embedding_model or os.getenv("RAG_EMBEDDING_MODEL", "")
        self.timeout = timeout
        self.retries = retries
        hostname = (urllib.parse.urlparse(self.base_url).hostname or "").casefold()
        if not self.api_key and hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("RAG_API_KEY is required for a non-local endpoint")

    def _post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            f"{self.base_url}/{endpoint.lstrip('/')}",
            data=body,
            headers=headers,
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                last_error = ProviderError(f"HTTP {exc.code}: {detail[:1000]}")
                if exc.code < 500 and exc.code != 429:
                    break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = ProviderError(str(exc))
            if attempt < self.retries:
                time.sleep(2**attempt)
        raise last_error or ProviderError("unknown provider error")

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not self.embedding_model:
            raise ValueError("RAG_EMBEDDING_MODEL is required")
        if not texts:
            return []
        response = self._post("embeddings", {"model": self.embedding_model, "input": texts})
        try:
            rows = sorted(response["data"], key=lambda item: item["index"])
            vectors = [row["embedding"] for row in rows]
        except (KeyError, TypeError) as exc:
            raise ProviderError("invalid embeddings response") from exc
        if len(vectors) != len(texts):
            raise ProviderError("embedding count does not match input count")
        return vectors

    def chat_json(
        self, messages: list[dict[str, str]], *, temperature: float = 0
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if not self.chat_model:
            raise ValueError("RAG_CHAT_MODEL is required")
        response = self._post(
            "chat/completions",
            {"model": self.chat_model, "messages": messages, "temperature": temperature,
             "response_format": {"type": "json_object"}},
        )
        try:
            content = response["choices"][0]["message"]["content"]
            result = json.loads(content)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise ProviderError("model did not return a valid JSON object") from exc
        return result, response.get("usage", {})
