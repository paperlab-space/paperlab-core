"""Keep the shipped examples and README snippets working."""

from __future__ import annotations

from pathlib import Path

from paperlab_space import ColumnMapping, Dataset, RunConfig, estimate, run
from paperlab_space.runner import prepare_data

from .fakes import FakeBackend

ROOT = Path(__file__).parent.parent


def test_sentiment_example() -> None:
    config = RunConfig.from_yaml(ROOT / "examples/sentiment/run.yaml")
    # Lift free-tier rate limits so the test does not wait in real time.
    unthrottled = [m.model_copy(update={"requests_per_minute": 1e6}) for m in config.models]
    config = config.model_copy(update={"models": unthrottled})
    data = prepare_data(config, None)
    assert len(data) == 12 and data.validate().issues == []
    results = run(config, data, backend=FakeBackend(lambda m, p, a: "Positive."))
    assert results.scores["Gemini 2.5 Flash"]["accuracy"].value == 0.5


def test_readme_python_snippet() -> None:
    data = Dataset.from_records([{"question": "2+2?", "answer": "4"}])
    config = RunConfig(
        mapping=ColumnMapping(input="question", expected="answer"),
        prompt="Answer briefly: {question}",
        models=["openrouter/nvidia/nemotron-3-super-120b-a12b:free", "gemini/gemini-2.5-flash"],
        metrics=["exact_match", "token_f1"],
    )
    assert estimate(config, data).models[0].cost_usd == 0.0
    results = run(config, data, backend=FakeBackend(lambda m, p, a: "4"))
    assert results.scores["gemini-2.5-flash"]["exact_match"].value == 1.0
