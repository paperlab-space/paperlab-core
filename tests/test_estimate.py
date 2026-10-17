from __future__ import annotations

import pytest

from paperlab_space.config import RunConfig
from paperlab_space.data import Dataset
from paperlab_space.runner import estimate
from paperlab_space.runner.estimate import approx_tokens

from .fakes import FakeBackend


def test_approx_tokens_by_script() -> None:
    assert approx_tokens("abcdefgh") == 2
    assert approx_tokens("你好世界") == 4
    assert approx_tokens("สวัสดี") == 6
    assert approx_tokens("") == 0


def test_estimate() -> None:
    data = Dataset.from_records([{"q": "a" * 40, "a": "b" * 8}] * 60)
    config = RunConfig.from_dict(
        {
            "mapping": {"input": "q", "expected": "a"},
            "prompt": "{q}",
            "models": ["openai/paid", "openrouter/x/y:free", "gemini/gemini-2.5-flash"],
        }
    )
    result = estimate(config, data, backend=FakeBackend(price=(1e-6, 2e-6)))
    paid, free, gemini = result.models
    assert paid.requests == 60
    assert paid.input_tokens == 60 * (10 + 8)
    assert paid.output_tokens == 60 * (3 + 16)
    assert paid.cost_usd == pytest.approx(60 * 18 * 1e-6 + 60 * 19 * 2e-6)
    assert free.minutes == pytest.approx(3.0)  # 60 requests at 20/min
    assert any("daily limit" in w for w in free.warnings)
    assert any("Google may use" in w for w in gemini.warnings)
    assert result.minutes == pytest.approx(6.0)  # gemini at 10/min is the slowest
    assert "Total" in str(result)


def test_unknown_price() -> None:
    data = Dataset.from_records([{"q": "x"}])
    config = RunConfig.from_dict(
        {
            "mapping": {"input": "q"},
            "prompt": "{q}",
            "models": [{"id": "openai/m", "max_tokens": 50}],
        }
    )
    result = estimate(config, data, backend=FakeBackend(price=None))
    assert result.cost_usd is None and result.models[0].output_tokens == 50


def test_reasoning_model_warning() -> None:
    data = Dataset.from_records([{"q": "x"}])
    config = RunConfig.from_dict(
        {
            "mapping": {"input": "q"},
            "prompt": "{q}",
            "models": ["gemini/gemini-2.5-flash", "openai/gpt-4o-mini"],
        }
    )
    thinking, plain = estimate(config, data).models
    assert any("reasoning model" in w for w in thinking.warnings)
    assert not any("reasoning model" in w for w in plain.warnings)
