from __future__ import annotations

import pytest
from pydantic import ValidationError

from paperlab_core.errors import TemplateError
from paperlab_core.prompt import PromptTemplate


def test_render_with_system_and_spaces() -> None:
    template = PromptTemplate(
        "Translate: {Source sentence}\nContext: { context }", system="You translate {lang}."
    )
    assert template.variables == ["Source sentence", "context", "lang"]
    messages = template.render({"Source sentence": "你好", "context": None, "lang": "zh"})
    assert messages == [
        {"role": "system", "content": "You translate zh."},
        {"role": "user", "content": "Translate: 你好\nContext: "},
    ]


def test_literal_braces() -> None:
    template = PromptTemplate('Return JSON like {{"label": "..."}} for: {text}')
    assert template.variables == ["text"]
    assert (
        template.render({"text": "hi"})[0]["content"] == 'Return JSON like {"label": "..."} for: hi'
    )


def test_check_columns() -> None:
    template = PromptTemplate("{a} {b}")
    template.check(["a", "b", "c"])
    with pytest.raises(TemplateError, match=r"\{b\}, which is not"):
        template.check(["a"])


def test_invalid_templates() -> None:
    with pytest.raises(ValidationError, match="no \\{column\\} placeholders"):
        PromptTemplate("No placeholders here")
    with pytest.raises(ValidationError, match="empty"):
        PromptTemplate("  ")


def test_from_string_in_model() -> None:
    assert PromptTemplate.model_validate("Q: {q}").user == "Q: {q}"
