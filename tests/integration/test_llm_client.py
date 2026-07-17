"""Real httpx over a real socket, against a stub server speaking Ollama's shape.

app/rag/llm.py is the least-covered module in the unit run (41%) because unit
tests replace the client wholesale. Here the actual client code executes: request
construction, JSON parsing, status handling, timeouts.
"""

import pytest

from app.core.errors import LLMUnavailable
from app.rag.llm import SYSTEM_PROMPT, OllamaLLM, build_prompt

from .conftest import StubOllama

pytestmark = pytest.mark.asyncio


async def test_completes_against_a_live_server():
    with StubOllama(reply="The notice period is 30 days [1].") as stub:
        llm = OllamaLLM(stub.url, "llama3.2:1b", timeout=10)
        out = await llm.complete(SYSTEM_PROMPT, "Question: notice period?")
    assert out == "The notice period is 30 days [1]."


async def test_sends_system_and_user_roles_in_order():
    with StubOllama() as stub:
        llm = OllamaLLM(stub.url, "llama3.2:1b", timeout=10)
        await llm.complete(SYSTEM_PROMPT, "user text")
    sent = stub.requests[0]
    assert [m["role"] for m in sent["messages"]] == ["system", "user"]
    assert sent["messages"][1]["content"] == "user text"


async def test_requests_non_streaming_and_low_temperature():
    """Streaming would break the client's single-shot JSON parse."""
    with StubOllama() as stub:
        llm = OllamaLLM(stub.url, "llama3.2:1b", timeout=10)
        await llm.complete("s", "u")
    sent = stub.requests[0]
    assert sent["stream"] is False
    assert sent["options"]["temperature"] == 0.1
    assert sent["model"] == "llama3.2:1b"


async def test_server_error_becomes_llm_unavailable():
    with StubOllama(status=500) as stub:
        llm = OllamaLLM(stub.url, "llama3.2:1b", timeout=10)
        with pytest.raises(LLMUnavailable):
            await llm.complete("s", "u")


async def test_connection_refused_becomes_llm_unavailable():
    from .conftest import free_port

    llm = OllamaLLM(f"http://127.0.0.1:{free_port()}", "llama3.2:1b", timeout=2)
    with pytest.raises(LLMUnavailable):
        await llm.complete("s", "u")


async def test_trailing_slash_in_url_does_not_double_up():
    with StubOllama() as stub:
        llm = OllamaLLM(stub.url + "/", "llama3.2:1b", timeout=10)
        out = await llm.complete("s", "u")
    assert out
    assert len(stub.requests) == 1


async def test_prompt_numbers_passages_for_citation():
    prompt = build_prompt("What is X?", ["first passage", "second passage"])
    assert "[1] first passage" in prompt
    assert "[2] second passage" in prompt
    assert "What is X?" in prompt


async def test_system_prompt_forbids_outside_knowledge():
    assert "ONLY" in SYSTEM_PROMPT
    assert "don't know" in SYSTEM_PROMPT.lower()
