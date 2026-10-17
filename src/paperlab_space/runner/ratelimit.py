"""Per-model rate limiting and retry backoff."""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable

from paperlab_space.config import ModelSpec, RetrySettings

Sleep = Callable[[float], Awaitable[None]]
Clock = Callable[[], float]

# Conservative defaults for free tiers; paid keys can override with requests_per_minute.
OPENROUTER_FREE_RPM = 20.0
GEMINI_FREE_RPM = 10.0
GEMINI_PRO_FREE_RPM = 5.0


def default_rpm(model: ModelSpec) -> float | None:
    """Requests per minute to stay under for ``model`` (``None`` = no limit)."""
    if model.requests_per_minute is not None:
        return model.requests_per_minute
    if model.provider == "openrouter" and model.id.endswith(":free"):
        return OPENROUTER_FREE_RPM
    if model.provider == "gemini":
        return GEMINI_PRO_FREE_RPM if "pro" in model.id.lower() else GEMINI_FREE_RPM
    return None


class RateLimiter:
    """Spaces requests evenly so no more than ``rpm`` start per minute.

    Even spacing (rather than bursts) is the safest way to stay inside free-tier limits.
    When a provider says "slow down", :meth:`pause` holds back every request to that model.
    """

    def __init__(
        self, rpm: float | None, clock: Clock = time.monotonic, sleep: Sleep = asyncio.sleep
    ) -> None:
        self.interval = 60.0 / rpm if rpm else 0.0
        self._clock = clock
        self._sleep = sleep
        self._next = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = self._clock()
            wait = max(0.0, self._next - now)
            self._next = max(now, self._next) + self.interval
        if wait > 0:
            await self._sleep(wait)

    def pause(self, seconds: float) -> None:
        self._next = max(self._next, self._clock() + seconds)


def backoff_delay(
    attempt: int,
    settings: RetrySettings,
    retry_after: float | None = None,
    rng: random.Random | None = None,
) -> float:
    """Seconds to wait before retry number ``attempt`` (1-based)."""
    if retry_after is not None:
        return min(max(retry_after, 0.0), settings.max_delay)
    ceiling = min(settings.max_delay, settings.base_delay * 2.0 ** (attempt - 1))
    return ceiling * (0.5 + 0.5 * (rng or random).random())
