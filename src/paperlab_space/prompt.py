"""Prompt templates with ``{column}`` placeholders."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator

from paperlab_space.errors import TemplateError

# ``{{`` and ``}}`` are literal braces; ``{name}`` is a placeholder. Column names may
# contain spaces and any non-brace characters, e.g. ``{Source sentence}``.
_TOKEN = re.compile(r"\{\{|\}\}|\{([^{}]+)\}")


class PromptTemplate(BaseModel):
    """A user prompt (and optional system prompt) with column placeholders.

    >>> PromptTemplate(user="Translate to English: {sentence}").render({"sentence": "สวัสดี"})
    [{'role': 'user', 'content': 'Translate to English: สวัสดี'}]
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    user: str
    system: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _from_string(cls, value: Any) -> Any:
        return {"user": value} if isinstance(value, str) else value

    def __init__(self, user: str | None = None, /, **data: Any) -> None:
        if user is not None:
            data["user"] = user
        super().__init__(**data)

    @model_validator(mode="after")
    def _check(self) -> PromptTemplate:
        if not self.user.strip():
            raise ValueError("The prompt template is empty.")
        if not self.variables:
            raise ValueError(
                "The prompt template has no {column} placeholders, so every row would get "
                "the same prompt."
            )
        return self

    @property
    def variables(self) -> list[str]:
        """Placeholder names in order of first appearance (user prompt, then system)."""
        names: list[str] = []
        for text in (self.user, self.system or ""):
            for match in _TOKEN.finditer(text):
                name = match.group(1)
                if name is not None and name.strip() not in names:
                    names.append(name.strip())
        return names

    def check(self, columns: Iterable[str]) -> None:
        """Raise :class:`TemplateError` if a placeholder is not one of ``columns``."""
        available = list(columns)
        missing = [v for v in self.variables if v not in available]
        if missing:
            raise TemplateError(
                f"The prompt uses {', '.join('{' + m + '}' for m in missing)}, which "
                f"{'is' if len(missing) == 1 else 'are'} not an input or context column. "
                f"Available: {', '.join('{' + c + '}' for c in available)}"
            )

    def render(self, fields: Mapping[str, str | None]) -> list[dict[str, str]]:
        """Fill the template and return chat messages. Empty cells become empty strings."""
        messages = []
        if self.system:
            messages.append({"role": "system", "content": _fill(self.system, fields)})
        messages.append({"role": "user", "content": _fill(self.user, fields)})
        return messages


def _fill(text: str, fields: Mapping[str, str | None]) -> str:
    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        if token == "{{":
            return "{"
        if token == "}}":
            return "}"
        name = match.group(1).strip()
        if name not in fields:
            raise TemplateError(f"No value for placeholder {{{name}}}.")
        return fields[name] or ""

    return _TOKEN.sub(replace, text)
