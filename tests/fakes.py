"""A scripted backend and a fake clock for testing the runner without network access."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable

from paperlab_space.config import ModelSpec
from paperlab_space.runner.backend import Completion, Messages, ProviderError, RetryableError


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


Responder = Callable[[ModelSpec, str, int], str | Exception]


class FakeBackend:
    """Answers with ``responder(model, user_prompt, attempt)``; exceptions are raised."""

    def __init__(
        self, responder: Responder | None = None, price: tuple[float, float] | None = (1e-6, 2e-6)
    ) -> None:
        self.responder = responder or (lambda model, prompt, attempt: prompt.upper())
        self.calls: list[tuple[str, str]] = []
        self.attempts: dict[tuple[str, str], int] = defaultdict(int)
        self._price = price

    async def complete(
        self, model: ModelSpec, messages: Messages, *, seed: int | None, timeout: float
    ) -> Completion:
        prompt = messages[-1]["content"]
        self.calls.append((model.id, prompt))
        self.attempts[(model.id, prompt)] += 1
        answer = self.responder(model, prompt, self.attempts[(model.id, prompt)])
        if isinstance(answer, Exception):
            raise answer
        return Completion(
            answer,
            input_tokens=len(prompt),
            output_tokens=len(answer),
            cost_usd=0.001,
            model_version=model.id + "-2026",
        )

    def price(self, model: ModelSpec) -> tuple[float, float] | None:
        return self._price


__all__ = ["FakeBackend", "FakeClock", "ProviderError", "RetryableError"]
