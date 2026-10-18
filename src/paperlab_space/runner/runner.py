"""Run every model over every record, respecting rate limits and retrying failures."""

from __future__ import annotations

import asyncio
import concurrent.futures
import datetime as dt
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from paperlab_space.config import ModelSpec, RunConfig
from paperlab_space.data.load import Dataset, load_dataset
from paperlab_space.data.mapping import MappedDataset
from paperlab_space.errors import ConfigError, DataError
from paperlab_space.results import ModelRun, RowOutput, RunResults
from paperlab_space.runner.backend import Backend, LiteLLMBackend, ProviderError, RetryableError
from paperlab_space.runner.ratelimit import Clock, RateLimiter, Sleep, backoff_delay, default_rpm


@dataclass
class Progress:
    """A progress update, sent after each row finishes."""

    model: str
    """The model's label."""
    done: int
    """Rows finished for this model, successful or not."""
    failed: int
    """Rows that failed so far for this model."""
    total: int
    """Rows to run for this model."""
    record_id: str
    """The row that just finished."""
    row: RowOutput | None = None
    """Its output, so callers can checkpoint partial results (see ``previous`` in :func:`arun`)."""


ProgressCallback = Callable[[Progress], None]

CHECKPOINT_INTERVAL_S = 5.0
"""Least time between two checkpoint writes."""


def run(
    config: RunConfig,
    data: Dataset | MappedDataset | None = None,
    *,
    backend: Backend | None = None,
    on_progress: ProgressCallback | None = None,
    previous: RunResults | None = None,
    checkpoint: str | Path | None = None,
) -> RunResults:
    """Run an evaluation and return outputs and scores.

    Works in scripts and in Jupyter (where an event loop is already running).
    See :func:`arun` for the arguments.
    """
    state = _Checkpoint(checkpoint, config, time.monotonic)
    coro = _arun(config, data, backend, on_progress, previous, state, asyncio.sleep, time.monotonic)
    try:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    except BaseException:
        # Ctrl+C can tear the event loop down before the coroutine sees it, so save here,
        # outside the loop, too.
        state.save()
        raise


async def arun(
    config: RunConfig,
    data: Dataset | MappedDataset | None = None,
    *,
    backend: Backend | None = None,
    on_progress: ProgressCallback | None = None,
    previous: RunResults | None = None,
    checkpoint: str | Path | None = None,
    sleep: Sleep = asyncio.sleep,
    clock: Clock = time.monotonic,
) -> RunResults:
    """Run an evaluation asynchronously.

    Args:
        config: What to run.
        data: The data. If omitted, it is loaded from ``config.dataset``. A plain
            :class:`Dataset` is mapped with ``config.mapping``.
        backend: Where to send requests. Defaults to :class:`LiteLLMBackend`.
        on_progress: Called after each row finishes.
        previous: Results of an earlier, interrupted run with the same config. Rows that
            already have an output are reused instead of requested again.
        checkpoint: A ``results.json`` path to save finished rows to while the run goes,
            and when it stops early (Ctrl+C, an error). Pass the file as ``previous`` to
            continue. When the run finishes, the full results are written there.
        sleep, clock: Injection points for tests.
    """
    state = _Checkpoint(checkpoint, config, clock)
    return await _arun(config, data, backend, on_progress, previous, state, sleep, clock)


async def _arun(
    config: RunConfig,
    data: Dataset | MappedDataset | None,
    backend: Backend | None,
    on_progress: ProgressCallback | None,
    previous: RunResults | None,
    state: _Checkpoint,
    sleep: Sleep,
    clock: Clock,
) -> RunResults:
    mapped = prepare_data(config, data)
    if config.metrics:
        from paperlab_space.metrics import check_metrics

        check_metrics(config, mapped)  # fail before spending any API calls
    backend = backend or LiteLLMBackend()
    state.start(mapped, _now())

    def track(progress: Progress) -> None:
        state.add(progress)
        if on_progress is not None:
            on_progress(progress)

    try:
        model_runs = await asyncio.gather(
            *(
                _run_model(config, model, mapped, backend, track, previous, sleep, clock)
                for model in config.models
            )
        )
    except GeneratorExit:  # the coroutine is being discarded; run() saves instead
        raise
    except BaseException:  # including Ctrl+C and cancellation: keep what is done
        state.save()
        raise
    results = RunResults(
        config=config, data=mapped, runs={r.name: r for r in model_runs}, started_at=state.started
    )
    if config.metrics:
        from paperlab_space.metrics import score

        results.scores = score(results)
    results.finished_at = _now()
    state.save(results)
    return results


class _Checkpoint:
    """The finished rows of a run, saved as ``results.json`` now and then (if a path is set)."""

    def __init__(self, path: str | Path | None, config: RunConfig, clock: Clock) -> None:
        self.path = Path(path) if path is not None else None
        self.config = config
        self.clock = clock
        self.rows: dict[str, dict[str, RowOutput]] = {m.name: {} for m in config.models}
        self.data: MappedDataset | None = None
        self.started: str | None = None
        self.last_save = clock()

    def start(self, data: MappedDataset, started: str) -> None:
        self.data, self.started = data, started

    def add(self, progress: Progress) -> None:
        if progress.row is not None:
            self.rows[progress.model][progress.record_id] = progress.row
        if self.path is not None and self.clock() - self.last_save >= CHECKPOINT_INTERVAL_S:
            self.save()
            self.last_save = self.clock()

    def save(self, results: RunResults | None = None) -> None:
        """Write ``results``, or the rows finished so far."""
        if self.path is None or self.data is None:
            return
        if results is None:
            runs = {
                m.name: ModelRun(m, list(self.rows[m.name].values())) for m in self.config.models
            }
            results = RunResults(
                config=self.config, data=self.data, runs=runs, started_at=self.started
            )
        # Write then rename, so an interruption mid-write never leaves a truncated file.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp")
        results.save_json(temporary)
        os.replace(temporary, self.path)


def prepare_data(config: RunConfig, data: Dataset | MappedDataset | None) -> MappedDataset:
    """Load and map the data for ``config``, refusing data with validation errors."""
    if data is None:
        if config.dataset is None:
            raise ConfigError("No data given and the config has no `dataset` section.")
        data = load_dataset(config.dataset.path, config.dataset.sheet)
    if isinstance(data, Dataset):
        mapped = data.map(config.mapping)
    else:
        if data.mapping != config.mapping:
            raise ConfigError("The data was mapped with a different ColumnMapping than the config.")
        mapped = data
    report = mapped.validate()
    if not report.ok:
        raise DataError(f"The data has problems that must be fixed first:\n{report}")
    return mapped


async def _run_model(
    config: RunConfig,
    model: ModelSpec,
    data: MappedDataset,
    backend: Backend,
    on_progress: ProgressCallback | None,
    previous: RunResults | None,
    sleep: Sleep,
    clock: Clock,
) -> ModelRun:
    limiter = RateLimiter(default_rpm(model), clock=clock, sleep=sleep)
    semaphore = asyncio.Semaphore(model.concurrency or config.concurrency)
    settings = config.retry
    seed = config.model_seed(model)
    total = len(data.records)
    rows: list[RowOutput | None] = [None] * total
    fatal: str | None = None
    done = failed = 0

    reusable: dict[str, RowOutput] = {}
    if previous is not None and model.name in previous.runs:
        if previous.runs[model.name].model != model:
            raise ConfigError(
                f"Cannot resume {model.name}: its settings changed since the last run."
            )
        reusable = {row.record_id: row for row in previous.runs[model.name].rows if row.ok}

    def finish(index: int, row: RowOutput) -> None:
        nonlocal done, failed
        rows[index] = row
        done += 1
        failed += not row.ok
        if on_progress is not None:
            on_progress(Progress(model.name, done, failed, total, row.record_id, row))

    async def one(index: int) -> None:
        nonlocal fatal
        record = data.records[index]
        if record.id in reusable:
            finish(index, reusable[record.id])
            return
        messages = config.prompt.render(record.fields)
        async with semaphore:
            attempts = 0
            while True:
                if fatal is not None:
                    finish(index, RowOutput(record.id, None, error=fatal, attempts=attempts))
                    return
                attempts += 1
                await limiter.acquire()
                if fatal is not None:  # another row failed fatally while this one waited
                    attempts -= 1
                    continue
                start = clock()
                try:
                    completion = await asyncio.wait_for(
                        backend.complete(model, messages, seed=seed, timeout=settings.timeout),
                        timeout=settings.timeout + 5,
                    )
                except (RetryableError, asyncio.TimeoutError) as exc:
                    retry_after = exc.retry_after if isinstance(exc, RetryableError) else None
                    if attempts >= settings.max_attempts:
                        message = str(exc) or "Timed out"
                        finish(index, RowOutput(record.id, None, error=message, attempts=attempts))
                        return
                    delay = backoff_delay(attempts, settings, retry_after)
                    if retry_after is not None:
                        limiter.pause(delay)
                    await sleep(delay)
                    continue
                except ProviderError as exc:
                    if exc.fatal:
                        fatal = str(exc)
                    finish(index, RowOutput(record.id, None, error=str(exc), attempts=attempts))
                    return
                except Exception as exc:  # a bug in a custom backend should not kill the run
                    finish(
                        index,
                        RowOutput(
                            record.id, None, error=f"{type(exc).__name__}: {exc}", attempts=attempts
                        ),
                    )
                    return
                finish(
                    index,
                    RowOutput(
                        record_id=record.id,
                        output=completion.text,
                        attempts=attempts,
                        latency_s=round(clock() - start, 3),
                        input_tokens=completion.input_tokens,
                        output_tokens=completion.output_tokens,
                        cost_usd=completion.cost_usd,
                        model_version=completion.model_version,
                    ),
                )
                return

    await asyncio.gather(*(one(i) for i in range(total)))
    return ModelRun(model=model, rows=[row for row in rows if row is not None])


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
