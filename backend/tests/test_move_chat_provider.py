import json

import httpx
import pytest

from backend.app.config import Settings
from backend.app.schemas.move_chat import ChatInput
from backend.app.services.move_chat import ChatProviderError, MoveChat


def test_move_chat_provider_sends_context_and_reads_changes(monkeypatch):
    chat = MoveChat(Settings(
        llm_base_url="https://test.invalid/v1", llm_api_key="mock-key", llm_model="gpt-6.2-sol",
    ))
    result = json.dumps({
        "message": "의자 수량을 바꿀까요?",
        "changes": {"items": [
            {"action": "update", "item_id": "chair", "changes": {"quantity": 3}},
        ]},
    })

    def reply(url, *, headers, json, timeout):
        assert url == "https://test.invalid/v1/chat/completions"
        assert headers == {"Authorization": "Bearer mock-key"}
        assert timeout == 30
        assert json["model"] == "gpt-6.2-sol"
        assert "chair" in json["messages"][1]["content"]
        assert json["messages"][-1]["content"] == "의자 3개로 바꿔줘"
        return httpx.Response(
            200, json={"choices": [{"message": {"content": result}}]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", reply)
    answer = chat.interpret(
        ChatInput(message="의자 3개로 바꿔줘"), {"items": [{"item_id": "chair"}]},
    )
    assert answer.changes.items[0].changes.quantity == 3


def test_move_chat_provider_rejects_malformed_model_output(monkeypatch):
    chat = MoveChat(Settings(
        llm_base_url="https://test.invalid/v1", llm_api_key="mock-key", llm_model="gpt-6.2-sol",
    ))

    def reply(url, **kwargs):
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "unexpected reply"}}]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", reply)
    with pytest.raises(ChatProviderError):
        chat.interpret(ChatInput(message="의자 3개로 바꿔줘"), {})
