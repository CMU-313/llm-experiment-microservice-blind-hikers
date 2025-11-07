from mock import Mock, patch  # type: ignore[import]

import sys
from pathlib import Path

try:  # pragma: no cover - optional IPython integration
    import ipytest  # type: ignore[import]
    from IPython import get_ipython  # type: ignore[import]
except Exception:  # noqa: BLE001 - best effort import for notebook flow
    ipytest = None
    get_ipython = lambda: None  # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.llm import client, query_llm_robust


def _maybe_enable_ipytest() -> None:
    shell = get_ipython()
    if ipytest is not None and shell is not None:  # pragma: no branch
        ipytest.autoconfig()


_maybe_enable_ipytest()


@patch.object(client, "chat")
def test_unexpected_language(mock_chat):
    mock_chat.return_value.message.content = "I don't understand your request"

    ok, text = query_llm_robust("Hier ist dein erstes Beispiel.")

    assert isinstance(ok, bool)
    assert isinstance(text, str)


@patch.object(client, "chat")
def test_model_raises_exception(mock_chat):
    mock_chat.side_effect = RuntimeError("LLM is down")

    ok, text = query_llm_robust("This should trigger an exception.")

    assert ok is False
    assert isinstance(text, str)
    assert len(text) > 0


@patch.object(client, "chat")
def test_non_english_flow(mock_chat):
    lang_response = Mock()
    lang_response.message.content = "German"

    translation_response = Mock()
    translation_response.message.content = "Here is your first example."

    mock_chat.side_effect = [lang_response, translation_response]

    ok, text = query_llm_robust("Hier ist dein erstes Beispiel.")

    assert ok is False
    assert "first example" in text
    assert mock_chat.call_count == 2


@patch.object(client, "chat")
def test_english_flow_short_circuits_translation(mock_chat):
    lang_response = Mock()
    lang_response.message.content = "English"
    mock_chat.return_value = lang_response

    post = "Here is your first example."
    ok, text = query_llm_robust(post)

    assert ok is True
    assert text == post
    assert mock_chat.call_count == 1


def _run_ipytest() -> None:
    if ipytest is None:
        raise SystemExit("ipytest is not available; install notebook deps to run")
    ipytest.run("-vv")


if __name__ == "__main__":
    _run_ipytest()

