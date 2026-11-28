import json

import httpx
import pytest

from backend.app.config import Settings
from backend.app.schemas.comparison_chat import ComparisonChatInput
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


def test_comparison_chat_provider_uses_the_confirmation_schema(monkeypatch):
    chat = MoveChat(Settings(
        llm_base_url="https://test.invalid/v1", llm_api_key="mock-key", llm_model="gpt-6.2-sol",
    ))
    items = [
        {"item_id": "chair", "label_ko": "의자", "result": "changed"},
        {"item_id": "wardrobe", "label_ko": "옷장", "result": "missing"},
    ]
    body = ComparisonChatInput(
        comparison_id="cmp_current", message="의자는 괜찮고 옷장은 아직 못 찾았어요.",
        history=[{"role": "assistant", "content": "무엇을 확인하셨어요?"}],
    )
    checks = [
        {"item_id": "chair", "user_check": "confirmed"},
        {"item_id": "wardrobe", "user_check": "not_found"},
    ]

    def reply(url, **kwargs):
        messages = kwargs["json"]["messages"]
        prompt, context, history, message = messages
        schema = json.loads(prompt["content"].split("응답 스키마: ", 1)[1])
        assert set(schema["properties"]) == {"message", "checks"}
        assert json.loads(context["content"].split(": ", 1)[1]) == {"items": items}
        assert history == body.history[0].model_dump()
        assert message["content"] == body.message
        return httpx.Response(200, json={
            "choices": [{"message": {
                "content": json.dumps({"message": "저장할까요?", "checks": checks}),
            }}],
        }, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", reply)
    assert [check.model_dump() for check in chat.interpret_comparison(body, items).checks] == checks


@pytest.mark.parametrize("checks", [
    [{"item_id": "chair", "user_check": "ok"}],
    [{"item_id": "chair", "user_check": "confirmed"}] * 2,
])
def test_comparison_chat_provider_rejects_invalid_checks(monkeypatch, checks):
    chat = MoveChat(Settings(
        llm_base_url="https://test.invalid/v1", llm_api_key="mock-key", llm_model="gpt-6.2-sol",
    ))

    def reply(url, **kwargs):
        return httpx.Response(200, json={
            "choices": [{"message": {
                "content": json.dumps({"message": "저장할까요?", "checks": checks}),
            }}],
        }, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", reply)
    body = ComparisonChatInput(comparison_id="cmp_current", message="의자 확인했어요.")
    with pytest.raises(ChatProviderError):
        chat.interpret_comparison(body, [])
