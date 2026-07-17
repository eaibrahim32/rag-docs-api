"""LLM clients behind one async interface.

Ollama is the default so the stack runs end-to-end with no API key and no spend;
OpenAI is a drop-in for production. Both go through the same prompt and the same
timeout, so switching backends is a config change, not a code change.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from functools import lru_cache

import httpx

from app.config import get_settings
from app.core.errors import LLMUnavailable

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a document question-answering assistant. Answer using ONLY the numbered "
    "context passages provided. Cite the passages you use inline as [1], [2], etc. "
    "If the context does not contain the answer, say you don't know based on the "
    "documents provided — do not use outside knowledge and do not guess."
)


def build_prompt(question: str, passages: list[str]) -> str:
    context = "\n\n".join(f"[{i}] {p}" for i, p in enumerate(passages, start=1))
    return f"Context passages:\n{context}\n\nQuestion: {question}\n\nAnswer:"


class LLM(ABC):
    name: str

    @abstractmethod
    async def complete(self, system: str, user: str) -> str: ...


class OllamaLLM(LLM):
    def __init__(self, url: str, model: str, timeout: float):
        self.url = url.rstrip("/")
        self.model = model
        self.name = f"ollama/{model}"
        self.timeout = timeout

    async def complete(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "options": {"temperature": 0.1},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(f"{self.url}/api/chat", json=payload)
                resp.raise_for_status()
                return resp.json()["message"]["content"].strip()
        except (httpx.HTTPError, KeyError) as exc:
            log.error("ollama call failed: %s", exc)
            raise LLMUnavailable(f"Ollama backend unreachable: {exc}") from exc


class OpenAILLM(LLM):
    def __init__(self, api_key: str, model: str, timeout: float):
        self.api_key = api_key
        self.model = model
        self.name = f"openai/{model}"
        self.timeout = timeout

    async def complete(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "temperature": 0.1,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    json=payload,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"].strip()
        except (httpx.HTTPError, KeyError) as exc:
            log.error("openai call failed: %s", exc)
            raise LLMUnavailable(f"OpenAI backend unreachable: {exc}") from exc


@lru_cache
def get_llm() -> LLM:
    settings = get_settings()
    if settings.llm_backend == "openai":
        if not settings.openai_api_key:
            raise LLMUnavailable("LLM_BACKEND=openai but OPENAI_API_KEY is unset")
        return OpenAILLM(settings.openai_api_key, settings.openai_model, settings.llm_timeout_s)
    return OllamaLLM(settings.ollama_url, settings.ollama_model, settings.llm_timeout_s)
