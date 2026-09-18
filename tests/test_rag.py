from app.nodes.classify import classify_node
from app.nodes.rewrite import rewrite_query_node


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class FakePromptChain:
    def __init__(self, content):
        self._content = content

    def invoke(self, *args, **kwargs):
        return FakeResponse(self._content)


def test_rewrite_keeps_original_question_when_llm_returns_answer(monkeypatch):
    monkeypatch.setattr("app.nodes.rewrite.get_llm", lambda: FakeLLM())
    monkeypatch.setattr(
        "app.nodes.rewrite.REWRITE_PROMPT",
        FakePromptChain(
            "Annual leave is paid time off from work that employees can use for vacation, personal matters, or rest."
        ),
    )

    result = rewrite_query_node({"original_question": "What is annual leave?"})

    assert result["question"] == "What is annual leave?"
    assert result["original_question"] == "What is annual leave?"


def test_classify_maps_faq_questions_to_policy(monkeypatch):
    monkeypatch.setattr("app.nodes.classify.get_llm", lambda: FakeLLM())
    monkeypatch.setattr("app.nodes.classify.CLASSIFY_PROMPT", FakePromptChain("faq"))

    result = classify_node({"question": "What is annual leave?"})

    assert result["query_type"] == "policy"
