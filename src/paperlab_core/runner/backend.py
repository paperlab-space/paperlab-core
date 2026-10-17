"""Model backends. The default backend talks to every provider through LiteLLM."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from paperlab_core.config import ModelSpec
from paperlab_core.errors import PaperLabError

Messages = list[dict[str, str]]


@dataclass
class Completion:
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    model_version: str | None = None
    """The exact model name reported by the provider, if any."""


class RetryableError(PaperLabError):
    """A transient failure (rate limit, timeout, server error) worth retrying."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class ProviderError(PaperLabError):
    """A permanent failure for one request.

    If ``fatal`` is true the problem affects every request to this model (bad API key,
    unknown model), so the runner stops sending requests to it.
    """

    def __init__(self, message: str, fatal: bool = False) -> None:
        super().__init__(message)
        self.fatal = fatal


class Backend(Protocol):
    async def complete(
        self, model: ModelSpec, messages: Messages, *, seed: int | None, timeout: float
    ) -> Completion:
        """Return one completion, raising :class:`RetryableError` or :class:`ProviderError`."""
        ...

    def price(self, model: ModelSpec) -> tuple[float, float] | None:
        """USD per input token and per output token, or ``None`` if unknown."""
        ...


class LiteLLMBackend:
    """Calls models through `LiteLLM <https://docs.litellm.ai>`_.

    API keys are read from the usual environment variables (``OPENROUTER_API_KEY``,
    ``GEMINI_API_KEY``, ``OPENAI_API_KEY``, ...) unless passed in ``api_keys``, keyed by
    provider prefix (e.g. ``{"openrouter": "sk-or-..."}``). Keys are never stored in the
    run config or results.
    """

    def __init__(
        self,
        api_keys: Mapping[str, str] | None = None,
        api_bases: Mapping[str, str] | None = None,
    ) -> None:
        self._api_keys = dict(api_keys or {})
        self._api_bases = dict(api_bases or {})

    async def complete(
        self, model: ModelSpec, messages: Messages, *, seed: int | None, timeout: float
    ) -> Completion:
        import litellm

        litellm.suppress_debug_info = True  # no "Give Feedback / Get Help" banner per error
        kwargs: dict[str, Any] = {
            "model": model.id,
            "messages": messages,
            "temperature": model.temperature,
            "timeout": timeout,
            "num_retries": 0,  # retries are handled by the runner
            "drop_params": True,  # silently drop params (e.g. seed) a provider does not support
        }
        optional = {
            "max_tokens": model.max_tokens,
            "top_p": model.top_p,
            "seed": seed,
            "api_key": self._api_keys.get(model.provider),
            "api_base": self._api_bases.get(model.provider),
        }
        kwargs.update({k: v for k, v in optional.items() if v is not None})
        kwargs.update(model.params)

        try:
            response = await litellm.acompletion(**kwargs)
        except Exception as exc:
            raise _translate(exc, model) from exc

        text = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", None)
        output_tokens = getattr(usage, "completion_tokens", None)
        try:
            cost: float | None = float(litellm.completion_cost(completion_response=response))
        except Exception:
            # LiteLLM does not know every model (e.g. OpenRouter free models); fall back to
            # our own price lookup.
            price = lookup_price(model)
            cost = None
            if price is not None and input_tokens is not None and output_tokens is not None:
                cost = input_tokens * price[0] + output_tokens * price[1]
        return Completion(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            model_version=getattr(response, "model", None),
        )

    def price(self, model: ModelSpec) -> tuple[float, float] | None:
        return lookup_price(model)


def lookup_price(model: ModelSpec) -> tuple[float, float] | None:
    """Look up per-token prices in LiteLLM's public price table."""
    if model.provider == "openrouter" and model.id.endswith(":free"):
        return (0.0, 0.0)
    if model.provider == "ollama":
        return (0.0, 0.0)
    import litellm

    parts = model.id.split("/")
    # Try the full ID, then progressively strip routing prefixes
    # (openrouter/openai/gpt-4o-mini -> openai/gpt-4o-mini -> gpt-4o-mini).
    for start in range(len(parts)):
        info = litellm.model_cost.get("/".join(parts[start:]))
        if info and "input_cost_per_token" in info:
            return float(info["input_cost_per_token"]), float(
                info.get("output_cost_per_token", 0.0)
            )
    return None


def is_reasoning_model(model: ModelSpec) -> bool:
    """Whether the model "thinks" before answering, billing hidden reasoning tokens."""
    import litellm

    try:
        return bool(litellm.supports_reasoning(model=model.id))
    except Exception:
        return False


def _translate(exc: Exception, model: ModelSpec) -> Exception:
    import litellm

    message = f"{type(exc).__name__}: {exc}"
    if isinstance(exc, litellm.AuthenticationError | litellm.PermissionDeniedError):
        return ProviderError(
            f"The {model.provider} API key was rejected. Check that it is correct and has "
            f"access to {model.id}. ({exc})",
            fatal=True,
        )
    if isinstance(exc, litellm.NotFoundError):
        return ProviderError(
            f"Model {model.id} was not found at {model.provider}. ({exc})", fatal=True
        )
    if isinstance(exc, litellm.RateLimitError):
        return RetryableError(message, retry_after=_retry_after(exc))
    if isinstance(
        exc,
        litellm.Timeout
        | litellm.APIConnectionError
        | litellm.ServiceUnavailableError
        | litellm.InternalServerError
        | litellm.BadGatewayError,
    ):
        return RetryableError(message, retry_after=_retry_after(exc))
    if isinstance(exc, litellm.BadRequestError):
        return ProviderError(message)
    if isinstance(exc, litellm.APIError) and getattr(exc, "status_code", 500) >= 500:
        return RetryableError(message)
    return ProviderError(message)


def _retry_after(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    value = headers.get("retry-after")
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None
