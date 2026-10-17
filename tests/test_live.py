"""Smoke tests against real providers. Run with: uv run pytest -m live

Each test is skipped unless the provider's API key is set. They send two tiny requests.
"""

from __future__ import annotations

import os

import pytest

from paperlab_space import Dataset, RunConfig, run

pytestmark = pytest.mark.live

MODELS = {
    "OPENROUTER_API_KEY": "openrouter/nvidia/nemotron-3-super-120b-a12b:free",
    "GEMINI_API_KEY": "gemini/gemini-3.5-flash",
}


@pytest.mark.parametrize("env_var", list(MODELS))
def test_provider(env_var: str) -> None:
    if not os.environ.get(env_var):
        pytest.skip(f"{env_var} not set")
    data = Dataset.from_records(
        [
            {"q": "What is 2 + 3? Answer with the number only.", "a": "5"},
            {"q": "ဆယ် ဆိုတာ ဂဏန်း ဘယ်လောက်လဲ။ Answer with the digits only.", "a": "10"},
        ]
    )
    config = RunConfig.from_dict(
        {
            "mapping": {"input": "q", "expected": "a"},
            "prompt": "{q}",
            "models": [{"id": MODELS[env_var], "max_tokens": 1024}],
            "metrics": ["exact_match"],
            "retry": {"max_attempts": 3},
        }
    )
    results = run(config, data)
    model_run = next(iter(results.runs.values()))
    errors = [row.error for row in model_run.rows if not row.ok]
    assert not errors, errors
    assert all(row.output for row in model_run.rows)
    assert results.scores[model_run.name]["exact_match"].value is not None
    print(
        model_run.name,
        [row.output for row in model_run.rows],
        results.scores[model_run.name]["exact_match"].value,
        model_run.model_versions,
    )
