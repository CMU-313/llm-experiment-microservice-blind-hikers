"""Utilities for interacting with the Basic LLM experiment client.

This module provides a configurable client wrapper plus helper functions that
implement a more robust query flow.  The functions are written with testability
in mind so that the surrounding application can rely on deterministic behaviour
even when the remote LLM service is unavailable.  When the optional ``ollama``
package is installed, a default client will be configured automatically using
environment variables that mirror the Basic LLM experiment notebooks.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, MutableMapping, Optional, Sequence


DEFAULT_MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

try:  # pragma: no cover - exercised indirectly via configure_client_from_env
    from ollama import Client as OllamaClient  # type: ignore
except Exception:  # noqa: BLE001 - broad to avoid importing issues at runtime
    OllamaClient = None  # type: ignore[assignment]


class PromptContext(Enum):
    """Reusable system prompts for the Basic LLM experiment."""

    CLASSIFICATION = (
        "You detect the language of the user's message. Reply with the language "
        "name in English (e.g. 'English', 'German')."
    )
    TRANSLATION = (
        "You translate non-English user messages into English. Reply only with "
        "the translated text."
    )


CLASSIFICATION_CONTEXT = PromptContext.CLASSIFICATION
TRANSLATION_CONTEXT = PromptContext.TRANSLATION


@dataclass
class ChatMessage:
    """Simple value object mirroring the OpenAI chat response interface."""

    role: str
    content: str


@dataclass
class ChatResponse:
    """Container for chat responses with a ``message`` attribute."""

    message: ChatMessage


class _UnconfiguredClient:
    """Fallback client used until a real client is registered."""

    def chat(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, str]],
        **_: MutableMapping[str, Any],
    ) -> ChatResponse:
        raise RuntimeError(
            "LLM client has not been configured. Call configure_client() with a "
            "client that implements a chat() method before using these helpers."
        )


client: Any = _UnconfiguredClient()
MODEL_NAME: str = DEFAULT_MODEL_NAME

if OllamaClient is not None:  # pragma: no cover - optional runtime path
    _default_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    try:
        client = OllamaClient(host=_default_host)
    except Exception:
        client = _UnconfiguredClient()



def configure_client(external_client: Any, *, model_name: Optional[str] = None) -> None:
    """Allow the application to inject a real chat client and optional model."""

    global client, MODEL_NAME
    client = external_client
    if model_name:
        MODEL_NAME = model_name


def configure_client_from_env(*, model_name: Optional[str] = None) -> None:
    """Configure an Ollama client using ``OLLAMA_HOST`` and optional model.

    This mirrors the quickstart code shared in the Basic LLM experiment
    notebooks, allowing the Flask app (or tests) to opt-in to a real model
    without importing Ollama in modules that do not need it.
    """

    if OllamaClient is None:
        raise RuntimeError(
            "ollama package is not installed; install it to configure a client"
        )

    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    configure_client(OllamaClient(host=host), model_name=model_name)


def _chat_with_system_prompt(*, system_prompt: str, user_content: str) -> ChatResponse:
    """Invoke the configured client while shielding callers from API details."""

    response = client.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )
    return response


def get_language(post: str) -> str:
    context = CLASSIFICATION_CONTEXT.value
    response = _chat_with_system_prompt(system_prompt=context, user_content=post)
    return response.message.content


def get_translation(post: str) -> str:
    context = TRANSLATION_CONTEXT.value
    response = _chat_with_system_prompt(system_prompt=context, user_content=post)
    return response.message.content


def query_llm(post: str) -> tuple[bool, str]:
    language = get_language(post)
    if language == "English":
        return True, post
    return False, get_translation(post)


def query_llm_robust(post: str) -> tuple[bool, str]:
    try:
        language = get_language(post)
    except Exception as exc:  # pragma: no cover - defensive; behaviour tested via mocks
        return False, f"Error: Language detection failed for '{post}' ({exc})"

    if not isinstance(language, str) or not language:
        return False, f"Error: Could not determine language for '{post}'"

    if language == "English":
        return True, post

    try:
        translation = get_translation(post)
    except Exception as exc:  # pragma: no cover - defensive; behaviour tested via mocks
        return False, f"Error: Translation failed for '{post}' ({exc})"

    if not isinstance(translation, str) or not translation:
        return False, f"Error: Could not translate '{post}'"

    return False, translation


__all__ = [
    "ChatMessage",
    "ChatResponse",
    "CLASSIFICATION_CONTEXT",
    "TRANSLATION_CONTEXT",
    "client",
    "configure_client",
    "configure_client_from_env",
    "get_language",
    "get_translation",
    "query_llm",
    "query_llm_robust",
]


