"""
generator.py

Synthesizes an answer from retrieved chunks using an LLM. The prompt
strictly instructs the model to only use the provided evidence, to tag
every claim with a [Source N] marker, and to say so explicitly if the
evidence is insufficient.

Providers supported (set via LLM_PROVIDER env var):
  - "anthropic"  (ANTHROPIC_API_KEY, model e.g. claude-sonnet-4-6)
  - "openai"     (OPENAI_API_KEY, model e.g. gpt-4o-mini)
  - "groq"       (GROQ_API_KEY, model e.g. llama-3.3-70b-versatile)
  - "mock"       (no key needed - extractive fallback, default)

Unlike a naive implementation, failures are never swallowed silently.
generate_answer() always returns a GenerationResult that reports:
  - whether generation actually succeeded with the requested provider
  - a machine-readable error_type (missing_api_key / authentication /
    permission / rate_limit / not_found / network / unknown)
  - a human-readable error_message
  - whether the answer shown is a mock fallback, so the UI/API can
    surface an honest status instead of pretending everything worked.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

from app.retriever import RetrievedChunk

NO_EVIDENCE_MESSAGE = (
    "The supplied sources do not contain enough information to answer this question."
)

SYSTEM_PROMPT = """You are a careful research assistant. You will be given a \
question and a list of numbered source excerpts. Answer ONLY using the \
information in the excerpts.

Rules:
- Every factual sentence in your answer must end with a citation marker like \
[Source 2], matching the excerpt number it came from.
- Combine information across sources where relevant.
- Do not use any outside knowledge and do not speculate beyond the excerpts.
- If the excerpts do not contain enough information to answer, respond with \
exactly: "{no_evidence}"
- Keep the answer concise (roughly 3-6 sentences or bullet points).
""".format(no_evidence=NO_EVIDENCE_MESSAGE)

REQUIRED_KEY_BY_PROVIDER = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "groq": "GROQ_API_KEY",
}


@dataclass
class GenerationResult:
    answer: str
    requested_provider: str
    provider_used: str
    success: bool
    used_fallback: bool
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    model: Optional[str] = None


class MissingAPIKeyError(Exception):
    pass


def _build_context(chunks: List[RetrievedChunk]) -> str:
    blocks = []
    for i, rc in enumerate(chunks, start=1):
        blocks.append(
            f"[Source {i}] (from {rc.chunk.doc_filename}, chunk #{rc.chunk.chunk_id})\n{rc.chunk.text}"
        )
    return "\n\n".join(blocks)


def _mock_generate(question: str, chunks: List[RetrievedChunk]) -> str:
    """Extractive, dependency-free fallback: pick the most relevant sentence
    from each retrieved chunk and tag it with its source number."""
    if not chunks:
        return NO_EVIDENCE_MESSAGE

    q_words = {w.lower() for w in re.findall(r"\w+", question) if len(w) > 3}
    lines = []
    for i, rc in enumerate(chunks, start=1):
        sentences = re.split(r"(?<=[.!?])\s+", rc.chunk.text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            continue
        best = max(
            sentences,
            key=lambda s: len(q_words & {w.lower() for w in re.findall(r"\w+", s)}),
        )
        lines.append(f"{best} [Source {i}]")

    if not lines:
        return NO_EVIDENCE_MESSAGE
    return "\n\n".join(lines)


def _require_key(provider: str) -> str:
    key_name = REQUIRED_KEY_BY_PROVIDER[provider]
    key = os.environ.get(key_name, "").strip()
    if not key:
        raise MissingAPIKeyError(
            f"{key_name} is not set. Add it to your .env file (see .env.example) "
            f"or export it in your shell before starting the app."
        )
    return key


def _anthropic_generate(question: str, chunks: List[RetrievedChunk]) -> tuple[str, str]:
    import anthropic

    _require_key("anthropic")
    model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    client = anthropic.Anthropic()
    context = _build_context(chunks)

    try:
        response = client.messages.create(
            model=model,
            max_tokens=800,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": f"Question: {question}\n\nExcerpts:\n{context}"}
            ],
        )
    except anthropic.AuthenticationError as e:
        raise RuntimeError(f"authentication:Anthropic rejected the API key ({e}).") from e
    except anthropic.PermissionDeniedError as e:
        raise RuntimeError(
            f"permission:This API key doesn't have access to model '{model}' ({e})."
        ) from e
    except anthropic.NotFoundError as e:
        raise RuntimeError(f"not_found:Model '{model}' was not found ({e}).") from e
    except anthropic.RateLimitError as e:
        raise RuntimeError(f"rate_limit:Anthropic rate limit or quota exceeded ({e}).") from e
    except anthropic.APIConnectionError as e:
        raise RuntimeError(f"network:Could not reach the Anthropic API ({e}).") from e

    text = "".join(block.text for block in response.content if block.type == "text")
    return text, model


def _openai_generate(question: str, chunks: List[RetrievedChunk]) -> tuple[str, str]:
    import openai
    from openai import OpenAI

    _require_key("openai")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    client = OpenAI()
    context = _build_context(chunks)

    try:
        response = client.chat.completions.create(
            model=model,
            max_tokens=800,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Question: {question}\n\nExcerpts:\n{context}"},
            ],
        )
    except openai.AuthenticationError as e:
        raise RuntimeError(f"authentication:OpenAI rejected the API key ({e}).") from e
    except openai.PermissionDeniedError as e:
        raise RuntimeError(
            f"permission:This API key doesn't have access to model '{model}' ({e})."
        ) from e
    except openai.NotFoundError as e:
        raise RuntimeError(f"not_found:Model '{model}' was not found ({e}).") from e
    except openai.RateLimitError as e:
        raise RuntimeError(f"rate_limit:OpenAI rate limit or quota exceeded ({e}).") from e
    except openai.APIConnectionError as e:
        raise RuntimeError(f"network:Could not reach the OpenAI API ({e}).") from e

    return response.choices[0].message.content, model


def _groq_generate(question: str, chunks: List[RetrievedChunk]) -> tuple[str, str]:
    import groq
    from groq import Groq

    _require_key("groq")
    model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
    client = Groq()
    context = _build_context(chunks)

    try:
        response = client.chat.completions.create(
            model=model,
            max_tokens=800,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Question: {question}\n\nExcerpts:\n{context}"},
            ],
        )
    except groq.AuthenticationError as e:
        raise RuntimeError(f"authentication:Groq rejected the API key ({e}).") from e
    except groq.PermissionDeniedError as e:
        raise RuntimeError(
            f"permission:This API key doesn't have access to model '{model}' ({e})."
        ) from e
    except groq.NotFoundError as e:
        raise RuntimeError(f"not_found:Model '{model}' was not found ({e}).") from e
    except groq.RateLimitError as e:
        raise RuntimeError(f"rate_limit:Groq rate limit or quota exceeded ({e}).") from e
    except groq.APIConnectionError as e:
        raise RuntimeError(f"network:Could not reach the Groq API ({e}).") from e

    return response.choices[0].message.content, model


_PROVIDERS = {
    "anthropic": _anthropic_generate,
    "openai": _openai_generate,
    "groq": _groq_generate,
}


def check_provider_status(provider: str) -> dict:
    """Lightweight, no-network check of whether a provider is ready to use
    (key present + matching SDK importable), for display in the UI before
    the user even asks a question."""
    provider = provider.lower()

    if provider == "mock" or provider not in _PROVIDERS:
        return {"provider": "mock", "ready": True, "detail": "No API key required."}

    key_name = REQUIRED_KEY_BY_PROVIDER[provider]
    key_present = bool(os.environ.get(key_name, "").strip())

    sdk_name = {"anthropic": "anthropic", "openai": "openai", "groq": "groq"}[provider]
    try:
        __import__(sdk_name)
        sdk_installed = True
    except ImportError:
        sdk_installed = False

    if not sdk_installed:
        return {
            "provider": provider,
            "ready": False,
            "detail": f"The '{sdk_name}' package is not installed. Run: pip install {sdk_name}",
        }
    if not key_present:
        return {
            "provider": provider,
            "ready": False,
            "detail": f"{key_name} is not set. Add it to your .env file.",
        }
    return {"provider": provider, "ready": True, "detail": f"{key_name} is set."}


def generate_answer(question: str, chunks: List[RetrievedChunk]) -> GenerationResult:
    """Always returns a GenerationResult - never raises, never silently
    lies about which provider actually produced the answer."""
    requested_provider = os.environ.get("LLM_PROVIDER", "mock").lower()

    if not chunks:
        return GenerationResult(
            answer=NO_EVIDENCE_MESSAGE,
            requested_provider=requested_provider,
            provider_used=requested_provider,
            success=True,
            used_fallback=False,
        )

    if requested_provider == "mock" or requested_provider not in _PROVIDERS:
        return GenerationResult(
            answer=_mock_generate(question, chunks),
            requested_provider=requested_provider,
            provider_used="mock",
            success=True,
            used_fallback=False,
        )

    fn = _PROVIDERS[requested_provider]

    try:
        answer, model = fn(question, chunks)
        return GenerationResult(
            answer=answer,
            requested_provider=requested_provider,
            provider_used=requested_provider,
            success=True,
            used_fallback=False,
            model=model,
        )
    except MissingAPIKeyError as e:
        return GenerationResult(
            answer=_mock_generate(question, chunks),
            requested_provider=requested_provider,
            provider_used="mock",
            success=False,
            used_fallback=True,
            error_type="missing_api_key",
            error_message=str(e),
        )
    except RuntimeError as e:
        msg = str(e)
        error_type, _, detail = msg.partition(":")
        if error_type not in {"authentication", "permission", "not_found", "rate_limit", "network"}:
            error_type, detail = "unknown", msg
        return GenerationResult(
            answer=_mock_generate(question, chunks),
            requested_provider=requested_provider,
            provider_used="mock",
            success=False,
            used_fallback=True,
            error_type=error_type,
            error_message=detail,
        )
    except Exception as e:  # noqa: BLE001 - final safety net, still surfaced to the user
        return GenerationResult(
            answer=_mock_generate(question, chunks),
            requested_provider=requested_provider,
            provider_used="mock",
            success=False,
            used_fallback=True,
            error_type="unknown",
            error_message=f"{type(e).__name__}: {e}",
        )
