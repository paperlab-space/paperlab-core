"""Run every model over every record, respecting rate limits and retrying failures."""

from __future__ import annotations

import asyncio
import concurrent.futures
import datetime as dt
import time
from collections.abc import Callable
from dataclasses import dataclass

from paperlab_core.config import ModelSpec, RunConfig
from paperlab_core.data.load import Dataset, load_dataset
from paperlab_core.data.mapping import MappedDataset
from paperlab_core.errors import ConfigError, DataError
from paperlab_core.results import ModelRun, RowOutput, RunResults
from paperlab_core.runner.backend import Backend, LiteLLMBackend, ProviderError, RetryableError
from paperlab_core.runner.ratelimit import Clock, RateLimiter, Sleep, backoff_delay, default_rpm


@dataclass
class Progress:
    model: str
    done: int
    """Rows finished for this model, successful or not."""
    failed: int
    total: int
    record_id: str
    """The row that just finished."""


ProgressCallback = Callable[[Progress], None]


def run(
    config: RunConfig,
    data: Dataset | MappedDataset | None = None,
    *,
    backend: Backend | None = None,
    on_progress: ProgressCallback | None = None,
    previous: RunResults | None = None,
) -> RunResults:
    """Run an evaluation and return outputs and scores.

    Works in scripts and in Jupyter (where an event loop is already running).
    See :func:`arun` for the arguments.
    """
    coro = arun(config, data, backend=backend, on_progress=on_progress, previous=previous)
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


async def arun(
    config: RunConfig,
    data: Dataset | MappedDataset | None = None,
    *,
    backend: Backend | None = None,
    on_progress: ProgressCallback | None = None,
    previous: RunResults | None = None,
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
        sleep, clock: Injection points for tests.
    """
    mapped = prepare_data(config, data)
    backend = backend or LiteLLMBackend()
    started = _now()
    model_runs = await asyncio.gather(
        *(
            _run_model(config, model, mapped, backend, on_progress, previous, sleep, clock)
            for model in config.models
        )
    )
    results = RunResults(
        config=config, data=mapped, runs={r.name: r for r in model_runs}, started_at=started
    )
    if config.metrics:
        from paperlab_core.metrics import score

        results.scores = score(results)
    results.finished_at = _now()
    return results


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
            on_progress(Progress(model.name, done, failed, total, row.record_id))

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
