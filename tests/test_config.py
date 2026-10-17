from __future__ import annotations

from pathlib import Path

import pytest

from paperlab_core.config import RunConfig
from paperlab_core.errors import ConfigError

YAML = """
dataset:
  path: data/test.xlsx
  sheet: test
mapping:
  input: sentence
  expected: gold
prompt: "Translate to English: {sentence}"
models:
  - openrouter/meta-llama/llama-3.3-70b-instruct:free
  - id: gemini/gemini-2.5-flash
    temperature: 0.7
    label: Gemini Flash
metrics:
  - exact_match
  - name: f1
    average: macro
seed: 42
"""


def test_yaml_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "run.yaml"
    path.write_text(YAML, encoding="utf-8")
    config = RunConfig.from_yaml(path)
    assert config.dataset is not None
    assert config.dataset.path == str((tmp_path / "data/test.xlsx").resolve())
    assert [m.name for m in config.models] == [
        "meta-llama/llama-3.3-70b-instruct:free",
        "Gemini Flash",
    ]
    assert config.models[0].provider == "openrouter"
    assert config.metrics[1].params == {"average": "macro"}
    assert config.model_seed(config.models[0]) == 42
    again = RunConfig.from_dict(__import__("yaml").safe_load(config.to_yaml()))
    assert again == config


def test_repro_file_is_accepted() -> None:
    config = RunConfig.from_dict(
        {
            "paperlab_core": "0.1.0",
            "config": {
                "mapping": {"input": "q"},
                "prompt": "{q}",
                "models": ["openai/gpt-4o-mini"],
            },
        }
    )
    assert config.models[0].id == "openai/gpt-4o-mini"


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"models": []}, "at least one model"),
        ({"models": ["gpt-4o"]}, "provider prefix"),
        ({"models": ["openai/gpt-4o", "openai/gpt-4o"]}, "distinct labels"),
        ({"prompt": "{missing}"}, "not an input or context column"),
        ({"temperatur": 1}, "temperatur"),
    ],
)
def test_invalid(override: dict[str, object], message: str) -> None:
    base: dict[str, object] = {"mapping": {"input": "q"}, "prompt": "{q}", "models": ["openai/x"]}
    with pytest.raises(ConfigError, match=message):
        RunConfig.from_dict({**base, **override})
