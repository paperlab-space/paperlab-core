"""Rough cost and time estimates shown before a run starts."""

from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass, field

from paperlab_space.config import RunConfig
from paperlab_space.data.load import Dataset
from paperlab_space.data.mapping import MappedDataset
from paperlab_space.runner.backend import Backend, is_reasoning_model, lookup_price
from paperlab_space.runner.ratelimit import default_rpm

# Daily caps worth warning about. OpenRouter limits ``:free`` models per day per account
# (lower for accounts that have never bought credits).
OPENROUTER_FREE_DAILY_REQUESTS = 50
DEFAULT_OUTPUT_TOKENS = 256
MESSAGE_OVERHEAD_TOKENS = 8
JUDGE_OUTPUT_TOKENS = 150
"""A short justification and the score line."""


@dataclass
class ModelEstimate:
    model: str
    requests: int
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    """``None`` when the model's price is unknown."""
    minutes: float
    warnings: list[str] = field(default_factory=list)


@dataclass
class Estimate:
    models: list[ModelEstimate]
    judges: list[ModelEstimate] = field(default_factory=list)
    """LLM-judge requests, one entry per judge metric."""

    @property
    def cost_usd(self) -> float | None:
        costs = [m.cost_usd for m in self.models + self.judges]
        return None if any(c is None for c in costs) else sum(c for c in costs if c is not None)

    @property
    def minutes(self) -> float:
        """Models run in parallel, so the slowest one sets the total. Judging follows."""
        models = max((m.minutes for m in self.models), default=0.0)
        return models + sum(j.minutes for j in self.judges)

    def __str__(self) -> str:
        lines = []
        for m in self.models + self.judges:
            cost = "unknown cost" if m.cost_usd is None else f"~${m.cost_usd:.4f}"
            lines.append(
                f"{m.model}: {m.requests} requests, ~{m.input_tokens:,} in / "
                f"~{m.output_tokens:,} out tokens, {cost}, ~{_fmt_minutes(m.minutes)}"
            )
            lines += [f"  ! {w}" for w in m.warnings]
        total = "unknown" if self.cost_usd is None else f"~${self.cost_usd:.4f}"
        lines.append(f"Total: {total}, ~{_fmt_minutes(self.minutes)}")
        return "\n".join(lines)


def estimate(
    config: RunConfig,
    data: Dataset | MappedDataset,
    *,
    backend: Backend | None = None,
    latency_s: float = 3.0,
) -> Estimate:
    """Estimate tokens, cost, and wall time for each model.

    Token counts use a script-aware heuristic, not the model's tokenizer, so treat them
    as ballpark figures. ``latency_s`` is the assumed time per request.
    """
    mapped = data.map(config.mapping) if isinstance(data, Dataset) else data
    requests = len(mapped.records)
    input_tokens = sum(
        sum(
            approx_tokens(m["content"]) + MESSAGE_OVERHEAD_TOKENS
            for m in config.prompt.render(r.fields)
        )
        for r in mapped.records
    )
    expected_tokens = [approx_tokens(r.expected[0]) for r in mapped.records if r.expected]

    estimates = []
    for model in config.models:
        if model.max_tokens is not None and not expected_tokens:
            per_row = model.max_tokens
        elif expected_tokens:
            # Models usually answer a bit longer than the reference.
            per_row = math.ceil(1.5 * sum(expected_tokens) / len(expected_tokens)) + 16
            if model.max_tokens is not None:
                per_row = min(per_row, model.max_tokens)
        else:
            per_row = DEFAULT_OUTPUT_TOKENS
        output_tokens = per_row * requests

        price = backend.price(model) if backend is not None else lookup_price(model)
        cost = None if price is None else input_tokens * price[0] + output_tokens * price[1]

        rpm = default_rpm(model)
        concurrency = model.concurrency or config.concurrency
        minutes = requests * latency_s / concurrency / 60
        if rpm:
            minutes = max(minutes, requests / rpm)

        warnings = []
        if price is None:
            warnings.append(f"No price found for {model.id}; cost is unknown.")
        if (
            model.provider == "openrouter"
            and model.id.endswith(":free")
            and requests > OPENROUTER_FREE_DAILY_REQUESTS
        ):
            warnings.append(
                f"{requests} requests may exceed OpenRouter's daily limit for free models "
                f"(as low as {OPENROUTER_FREE_DAILY_REQUESTS}/day without purchased credits). "
                "The run will retry, but may not finish today."
            )
        if is_reasoning_model(model):
            warnings.append(
                "This is a reasoning model: hidden thinking tokens are billed as output and can "
                "make the real cost several times higher than this estimate. Set max_tokens, or "
                "a provider option such as reasoning_effort, to cap it."
            )
        if model.provider == "gemini":
            warnings.append(
                "If your Gemini key is on the free tier, Google may use the "
                "submitted data to improve its products."
            )
        estimates.append(
            ModelEstimate(
                model.name, requests, input_tokens, output_tokens, cost, round(minutes, 2), warnings
            )
        )
    return Estimate(estimates, _judge_estimates(config, mapped, estimates, backend, latency_s))


def _judge_estimates(
    config: RunConfig,
    data: MappedDataset,
    models: list[ModelEstimate],
    backend: Backend | None,
    latency_s: float,
) -> list[ModelEstimate]:
    from paperlab_space.metrics import LLMJudge, get_metric

    estimates = []
    evaluated = {m.id for m in config.models}
    for spec in config.metrics:
        metric = get_metric(spec)
        if not isinstance(metric, LLMJudge):
            continue
        use_reference = metric.uses_reference(data)
        records = [r for r in data.records if r.expected or not use_reference]
        # The judge prompt holds the task, the reference, the rubric, and the model's output.
        base = sum(
            sum(
                approx_tokens(m["content"])
                for m in metric.messages(
                    config.prompt.render(r.fields)[-1]["content"],
                    "",
                    r.expected if use_reference else None,
                )
            )
            + MESSAGE_OVERHEAD_TOKENS * 2
            for r in records
        )
        input_tokens = output_tokens = 0
        for m in models:
            per_output = m.output_tokens // max(m.requests, 1)
            input_tokens += base + per_output * len(records)
        requests = len(records) * len(models)
        judge = metric.model
        if judge.max_tokens is not None:
            output_tokens = min(JUDGE_OUTPUT_TOKENS, judge.max_tokens) * requests
        else:
            output_tokens = JUDGE_OUTPUT_TOKENS * requests
        price = backend.price(judge) if backend is not None else lookup_price(judge)
        cost = None if price is None else input_tokens * price[0] + output_tokens * price[1]
        minutes = requests * latency_s / (judge.concurrency or config.concurrency) / 60
        rpm = default_rpm(judge)
        if rpm:
            minutes = max(minutes, requests / rpm)
        warnings = []
        if price is None:
            warnings.append(f"No price found for {judge.id}; cost is unknown.")
        if judge.id in evaluated:
            warnings.append(
                f"{judge.id} is also one of the models being evaluated. LLM judges tend to "
                "favour their own answers; consider a judge from another model family."
            )
        if is_reasoning_model(judge):
            warnings.append(
                "The judge is a reasoning model: hidden thinking tokens can make it cost "
                "several times this estimate."
            )
        estimates.append(
            ModelEstimate(
                f"{judge.name} (judge for {spec.key})",
                requests,
                input_tokens,
                output_tokens,
                cost,
                round(minutes, 2),
                warnings,
            )
        )
    return estimates


def approx_tokens(text: str) -> int:
    """Approximate token count: ~4 characters per token for Latin-like scripts, about one
    token per character for scripts written without spaces (CJK, Thai, Burmese, ...)."""
    dense = sum(1 for ch in text if _is_dense(ch))
    return dense + math.ceil((len(text) - dense) / 4)


def _is_dense(ch: str) -> bool:
    if ch.isascii():
        return False
    name = unicodedata.name(ch, "")
    return name.startswith(
        (
            "CJK",
            "HIRAGANA",
            "KATAKANA",
            "HANGUL",
            "THAI",
            "MYANMAR",
            "LAO",
            "KHMER",
            "TIBETAN",
            "DEVANAGARI",
            "BENGALI",
            "TAMIL",
            "SINHALA",
        )
    )


def _fmt_minutes(minutes: float) -> str:
    if minutes < 1:
        return f"{max(1, round(minutes * 60))} s"
    if minutes < 90:
        return f"{minutes:.0f} min"
    return f"{minutes / 60:.1f} h"
