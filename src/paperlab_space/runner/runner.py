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
from typing import TYPE_CHECKING

from paperlab_space.config import MetricSpec, ModelSpec, RunConfig
from paperlab_space.data.load import Dataset, load_dataset
from paperlab_space.data.mapping import MappedDataset, Record
from paperlab_space.errors import ConfigError, DataError
from paperlab_space.results import ModelRun, RowOutput, RunResults
from paperlab_space.runner.backend import (
    Backend,
    LiteLLMBackend,
    Messages,
    ProviderError,
    RetryableError,
)
from paperlab_space.runner.ratelimit import Clock, RateLimiter, Sleep, backoff_delay, default_rpm

if TYPE_CHECKING:
    from paperlab_space.metrics.judge import LLMJudge


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
    metric: str | None = None
    """For LLM-judge requests, the judge metric's key. ``model`` is then the model whose
    output was judged, and ``done`` and ``total`` count this metric's judge requests."""


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
    on_judge_progress: ProgressCallback | None = None,
) -> RunResults:
    """Run an evaluation and return outputs and scores.

    Works in scripts and in Jupyter (where an event loop is already running).
    See :func:`arun` for the arguments.
    """
    state = _Checkpoint(checkpoint, config, time.monotonic)
    coro = _arun(
        config,
        data,
        backend,
        _Callbacks(on_progress, on_judge_progress),
        previous,
        state,
        asyncio.sleep,
        time.monotonic,
    )
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
    on_judge_progress: ProgressCallback | None = None,
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
        previous: Results of an earlier, interrupted run with the same models. Rows that
            already have an output are reused instead of requested again, and so are LLM-judge
            responses for unchanged outputs and judge settings.
        checkpoint: A ``results.json`` path to save finished rows to while the run goes,
            and when it stops early (Ctrl+C, an error). Pass the file as ``previous`` to
            continue. When the run finishes, the full results are written there.
        on_judge_progress: Called after each LLM-judge request finishes. Judging starts once
            every model has finished.
        sleep, clock: Injection points for tests.
    """
    state = _Checkpoint(checkpoint, config, clock)
    callbacks = _Callbacks(on_progress, on_judge_progress)
    return await _arun(config, data, backend, callbacks, previous, state, sleep, clock)


@dataclass
class _Callbacks:
    model: ProgressCallback | None
    judge: ProgressCallback | None


async def _arun(
    config: RunConfig,
    data: Dataset | MappedDataset | None,
    backend: Backend | None,
    callbacks: _Callbacks,
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
        if callbacks.model is not None:
            callbacks.model(progress)

    def track_judge(progress: Progress) -> None:
        state.add(progress)
        if callbacks.judge is not None:
            callbacks.judge(progress)

    try:
        model_runs = await asyncio.gather(
            *(
                _run_model(config, model, mapped, backend, track, previous, sleep, clock)
                for model in config.models
            )
        )
        runs = {r.name: r for r in model_runs}
        judgments = await _judge_all(
            config, mapped, runs, backend, track_judge, previous, sleep, clock
        )
    except GeneratorExit:  # the coroutine is being discarded; run() saves instead
        raise
    except BaseException:  # including Ctrl+C and cancellation: keep what is done
        state.save()
        raise
    results = RunResults(
        config=config, data=mapped, runs=runs, judgments=judgments, started_at=state.started
    )
    if config.metrics:
        from paperlab_space.metrics import score
        from paperlab_space.stats import compare

        results.scores = score(results)
        results.comparisons = compare(results)
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
        self.judgments: dict[str, dict[str, dict[str, RowOutput]]] = {}
        self.data: MappedDataset | None = None
        self.started: str | None = None
        self.last_save = clock()

    def start(self, data: MappedDataset, started: str) -> None:
        self.data, self.started = data, started

    def add(self, progress: Progress) -> None:
        if progress.row is not None and progress.metric is not None:
            judged = self.judgments.setdefault(progress.metric, {}).setdefault(progress.model, {})
            judged[progress.record_id] = progress.row
        elif progress.row is not None:
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
            judgments = {
                key: {name: list(rows.values()) for name, rows in models.items()}
                for key, models in self.judgments.items()
            }
            results = RunResults(
                config=self.config,
                data=self.data,
                runs=runs,
                judgments=judgments,
                started_at=self.started,
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
    reusable: dict[str, RowOutput] = {}
    if previous is not None and model.name in previous.runs:
        if previous.runs[model.name].model != model:
            raise ConfigError(
                f"Cannot resume {model.name}: its settings changed since the last run."
            )
        reusable = {row.record_id: row for row in previous.runs[model.name].rows if row.ok}

    total = len(data.records)
    done = failed = 0

    def finish(index: int, row: RowOutput) -> None:
        nonlocal done, failed
        done += 1
        failed += not row.ok
        if on_progress is not None:
            on_progress(Progress(model.name, done, failed, total, row.record_id, row))

    def request(record: Record) -> Callable[[], Messages]:
        return lambda: config.prompt.render(record.fields)

    jobs: list[Job] = [(record.id, request(record)) for record in data.records]
    reuse = {i: reusable[r.id] for i, r in enumerate(data.records) if r.id in reusable}
    rows = await _run_requests(
        config, model, config.model_seed(model), jobs, reuse, finish, backend, sleep, clock
    )
    return ModelRun(model=model, rows=rows)


Job = tuple[str, Callable[[], Messages]]
"""A record ID and a function returning the messages to send for it."""


async def _run_requests(
    config: RunConfig,
    model: ModelSpec,
    seed: int | None,
    jobs: list[Job],
    reusable: dict[int, RowOutput],
    finish: Callable[[int, RowOutput], None],
    backend: Backend,
    sleep: Sleep,
    clock: Clock,
) -> list[RowOutput]:
    """Send every job to ``model``, with rate limiting and retries, in job order."""
    limiter = RateLimiter(default_rpm(model), clock=clock, sleep=sleep)
    semaphore = asyncio.Semaphore(model.concurrency or config.concurrency)
    settings = config.retry
    rows: list[RowOutput | None] = [None] * len(jobs)
    fatal: str | None = None

    def done(index: int, row: RowOutput) -> None:
        rows[index] = row
        finish(index, row)

    async def one(index: int) -> None:
        nonlocal fatal
        record_id, build = jobs[index]
        if index in reusable:
            done(index, reusable[index])
            return
        messages = build()
        async with semaphore:
            attempts = 0
            while True:
                if fatal is not None:
                    done(index, RowOutput(record_id, None, error=fatal, attempts=attempts))
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
                        done(index, RowOutput(record_id, None, error=message, attempts=attempts))
                        return
                    delay = backoff_delay(attempts, settings, retry_after)
                    if retry_after is not None:
                        limiter.pause(delay)
                    await sleep(delay)
                    continue
                except ProviderError as exc:
                    if exc.fatal:
                        fatal = str(exc)
                    done(index, RowOutput(record_id, None, error=str(exc), attempts=attempts))
                    return
                except Exception as exc:  # a bug in a custom backend should not kill the run
                    done(
                        index,
                        RowOutput(
                            record_id, None, error=f"{type(exc).__name__}: {exc}", attempts=attempts
                        ),
                    )
                    return
                done(
                    index,
                    RowOutput(
                        record_id=record_id,
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

    await asyncio.gather(*(one(i) for i in range(len(jobs))))
    return [row for row in rows if row is not None]


async def _judge_all(
    config: RunConfig,
    data: MappedDataset,
    runs: dict[str, ModelRun],
    backend: Backend,
    on_progress: ProgressCallback,
    previous: RunResults | None,
    sleep: Sleep,
    clock: Clock,
) -> dict[str, dict[str, list[RowOutput]]]:
    """Run every LLM-judge metric over every model's outputs, one metric at a time."""
    from paperlab_space.metrics import LLMJudge, get_metric

    judgments: dict[str, dict[str, list[RowOutput]]] = {}
    for spec in config.metrics:
        metric = get_metric(spec)
        if isinstance(metric, LLMJudge):
            judgments[spec.key] = await _judge(
                config, spec, metric, data, runs, backend, on_progress, previous, sleep, clock
            )
    return judgments


async def _judge(
    config: RunConfig,
    spec: MetricSpec,
    metric: LLMJudge,
    data: MappedDataset,
    runs: dict[str, ModelRun],
    backend: Backend,
    on_progress: ProgressCallback,
    previous: RunResults | None,
    sleep: Sleep,
    clock: Clock,
) -> dict[str, list[RowOutput]]:
    from paperlab_space.metrics.judge import NOT_JUDGED

    use_reference = metric.uses_reference(data)
    earlier = _earlier_judgments(previous, spec)
    out = {name: [RowOutput(r.id, None, error=NOT_JUDGED) for r in data.records] for name in runs}

    def request(record: Record, output: str) -> Callable[[], Messages]:
        def build() -> Messages:
            task = config.prompt.render(record.fields)[-1]["content"]
            return metric.messages(task, output, record.expected if use_reference else None)

        return build

    jobs: list[Job] = []
    slots: list[tuple[str, int]] = []
    reuse: dict[int, RowOutput] = {}
    for name, model_run in runs.items():
        for index, (record, row) in enumerate(zip(data.records, model_run.rows, strict=True)):
            if row.output is None or (use_reference and not record.expected):
                continue
            old = earlier.get((name, record.id))
            if old is not None and old[0] == row.output:
                reuse[len(jobs)] = old[1]
            jobs.append((record.id, request(record, row.output)))
            slots.append((name, index))

    total = len(jobs)
    done = failed = 0

    def finish(job: int, row: RowOutput) -> None:
        nonlocal done, failed
        name, index = slots[job]
        out[name][index] = row
        done += 1
        failed += not row.ok
        on_progress(Progress(name, done, failed, total, row.record_id, row, metric=spec.key))

    judge = metric.model
    seed = judge.seed if judge.seed is not None else config.seed
    await _run_requests(config, judge, seed, jobs, reuse, finish, backend, sleep, clock)
    return out


def _earlier_judgments(
    previous: RunResults | None, spec: MetricSpec
) -> dict[tuple[str, str], tuple[str, RowOutput]]:
    """(model, record ID) -> (the output that was judged, the judgment), reusable when the
    judge metric's settings are unchanged."""
    if previous is None or spec.key not in previous.judgments:
        return {}
    if not any(s == spec for s in previous.config.metrics):
        return {}
    reusable = {}
    for name, rows in previous.judgments[spec.key].items():
        model_run = previous.runs.get(name)
        if model_run is None:
            continue
        outputs = {row.record_id: row.output for row in model_run.rows}
        for row in rows:
            output = outputs.get(row.record_id)
            if row.ok and output is not None:
                reusable[(name, row.record_id)] = (output, row)
    return reusable


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
