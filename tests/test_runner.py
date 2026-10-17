from __future__ import annotations

from pathlib import Path

import pytest

from paperlab_space.config import ModelSpec, RetrySettings, RunConfig
from paperlab_space.data import ColumnMapping, Dataset
from paperlab_space.errors import ConfigError, DataError
from paperlab_space.results import RunResults
from paperlab_space.runner import Progress, arun, run
from paperlab_space.runner.ratelimit import RateLimiter, backoff_delay, default_rpm

from .fakes import FakeBackend, FakeClock, ProviderError, RetryableError

DATA = Dataset.from_records([{"id": f"r{i}", "q": f"q{i}", "a": f"Q{i}"} for i in range(1, 6)])


def make_config(**overrides: object) -> RunConfig:
    base: dict[str, object] = {
        "mapping": {"input": "q", "expected": "a", "id": "id"},
        "prompt": "{q}",
        "models": ["openai/model-a", "openai/model-b"],
        "retry": {"max_attempts": 3, "base_delay": 1.0},
    }
    return RunConfig.from_dict({**base, **overrides})


async def test_runs_all_models_and_rows() -> None:
    clock = FakeClock()
    progress: list[Progress] = []
    results = await arun(
        make_config(),
        DATA,
        backend=FakeBackend(),
        on_progress=progress.append,
        sleep=clock.sleep,
        clock=clock,
    )
    assert results.model_names == ["model-a", "model-b"]
    run_a = results.runs["model-a"]
    assert [r.output for r in run_a.rows] == ["Q1", "Q2", "Q3", "Q4", "Q5"]
    assert [r.record_id for r in run_a.rows] == ["r1", "r2", "r3", "r4", "r5"]
    assert run_a.n_failed == 0 and run_a.cost_usd == pytest.approx(0.005)
    assert run_a.model_versions == ["openai/model-a-2026"]
    assert len(progress) == 10 and {p.total for p in progress} == {5}
    assert max(p.done for p in progress if p.model == "model-b") == 5
    assert results.started_at and results.finished_at


async def test_retries_then_succeeds() -> None:
    clock = FakeClock()

    def flaky(model: ModelSpec, prompt: str, attempt: int) -> str | Exception:
        return RetryableError("429 slow down") if attempt < 3 else "ok"

    results = await arun(
        make_config(models=["openai/m"]),
        DATA,
        backend=FakeBackend(flaky),
        sleep=clock.sleep,
        clock=clock,
    )
    rows = results.runs["m"].rows
    assert all(r.output == "ok" and r.attempts == 3 for r in rows)
    assert len(clock.sleeps) == 10  # two backoffs per row


async def test_gives_up_after_max_attempts() -> None:
    clock = FakeClock()
    backend = FakeBackend(lambda m, p, a: RetryableError("server error"))
    results = await arun(
        make_config(models=["openai/m"]), DATA, backend=backend, sleep=clock.sleep, clock=clock
    )
    rows = results.runs["m"].rows
    assert all(r.output is None and r.error == "server error" and r.attempts == 3 for r in rows)
    assert results.runs["m"].n_failed == 5


async def test_fatal_error_stops_model_but_not_others() -> None:
    clock = FakeClock()

    def responder(model: ModelSpec, prompt: str, attempt: int) -> str | Exception:
        if model.id == "openai/bad":
            return ProviderError("API key rejected", fatal=True)
        return "fine"

    backend = FakeBackend(responder)
    config = make_config(models=["openai/bad", "openai/good"], concurrency=1)
    results = await arun(config, DATA, backend=backend, sleep=clock.sleep, clock=clock)
    assert sum(1 for model, _ in backend.calls if model == "openai/bad") == 1
    assert all(r.error == "API key rejected" for r in results.runs["bad"].rows)
    assert all(r.output == "fine" for r in results.runs["good"].rows)


async def test_non_fatal_error_only_affects_row() -> None:
    clock = FakeClock()
    backend = FakeBackend(lambda m, p, a: ProviderError("too long") if p == "q2" else "ok")
    results = await arun(
        make_config(models=["openai/m"]), DATA, backend=backend, sleep=clock.sleep, clock=clock
    )
    assert [r.ok for r in results.runs["m"].rows] == [True, False, True, True, True]


async def test_resume_reuses_successful_rows(tmp_path: Path) -> None:
    clock = FakeClock()
    first = await arun(
        make_config(models=["openai/m"]),
        DATA,
        backend=FakeBackend(lambda m, p, a: ProviderError("x") if p == "q3" else "ok"),
        sleep=clock.sleep,
        clock=clock,
    )
    path = tmp_path / "results.json"
    first.save_json(path)
    loaded = RunResults.load_json(path)
    backend = FakeBackend(lambda m, p, a: "new")
    second = await arun(
        loaded.config, loaded.data, backend=backend, previous=loaded, sleep=clock.sleep, clock=clock
    )
    assert backend.calls == [("openai/m", "q3")]
    assert [r.output for r in second.runs["m"].rows] == ["ok", "ok", "new", "ok", "ok"]


async def test_resume_refuses_changed_model() -> None:
    clock = FakeClock()
    first = await arun(
        make_config(models=["openai/m"]),
        DATA,
        backend=FakeBackend(),
        sleep=clock.sleep,
        clock=clock,
    )
    changed = make_config(models=[{"id": "openai/m", "temperature": 1.0}])
    with pytest.raises(ConfigError, match="settings changed"):
        await arun(
            changed, DATA, backend=FakeBackend(), previous=first, sleep=clock.sleep, clock=clock
        )


def test_sync_run_and_bad_data() -> None:
    results = run(make_config(models=["openai/m"]), DATA, backend=FakeBackend())
    assert results.runs["m"].rows[0].output == "Q1"
    bad = Dataset.from_records([{"id": "x", "q": "1", "a": "1"}, {"id": "x", "q": "2", "a": "2"}])
    with pytest.raises(DataError, match="duplicate"):
        run(make_config(), bad, backend=FakeBackend())
    with pytest.raises(ConfigError, match="No data"):
        run(make_config(), None, backend=FakeBackend())
    other = DATA.map(ColumnMapping(input="q", id="id"))
    with pytest.raises(ConfigError, match="different ColumnMapping"):
        run(make_config(), other, backend=FakeBackend())


async def test_sync_run_inside_event_loop() -> None:
    # e.g. Jupyter, where a loop is already running
    results = run(make_config(models=["openai/m"]), DATA, backend=FakeBackend())
    assert len(results.runs["m"].rows) == 5


async def test_rate_limiter_spacing() -> None:
    clock = FakeClock()
    limiter = RateLimiter(rpm=30, clock=clock, sleep=clock.sleep)
    for _ in range(3):
        await limiter.acquire()
    assert clock.sleeps == [2.0, 2.0]
    limiter.pause(10)
    await limiter.acquire()
    assert clock.sleeps[-1] == pytest.approx(10.0)


def test_default_rpm_and_backoff() -> None:
    assert default_rpm(ModelSpec(id="openrouter/x/y:free")) == 20
    assert default_rpm(ModelSpec(id="gemini/gemini-2.5-pro")) == 5
    assert default_rpm(ModelSpec(id="gemini/gemini-2.5-flash")) == 10
    assert default_rpm(ModelSpec(id="openai/gpt-4o")) is None
    assert default_rpm(ModelSpec(id="openai/gpt-4o", requests_per_minute=7)) == 7
    settings = RetrySettings(base_delay=2, max_delay=10)
    assert 1.0 <= backoff_delay(1, settings) <= 2.0
    assert 5.0 <= backoff_delay(9, settings) <= 10.0
    assert backoff_delay(1, settings, retry_after=30) == 10
