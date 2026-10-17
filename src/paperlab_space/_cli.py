"""The ``paperlab-space`` command."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
import yaml
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeRemainingColumn
from rich.table import Table

from paperlab_space import __version__
from paperlab_space.config import RunConfig
from paperlab_space.data import ColumnMapping, list_sheets, load_dataset
from paperlab_space.errors import PaperLabError
from paperlab_space.export import write_all
from paperlab_space.metrics import available_metrics
from paperlab_space.results import RunResults
from paperlab_space.runner import Progress as RunProgress
from paperlab_space.runner import estimate, prepare_data, run

app = typer.Typer(
    help="Evaluate LLMs on spreadsheet data and get paper-ready results.",
    no_args_is_help=True,
    add_completion=False,
)
# emoji=False: model IDs like `x:free:` must not turn into emoji.
console = Console(emoji=False)
err = Console(stderr=True, emoji=False)

_EXPECTED_HINTS = ("expected", "gold", "answer", "label", "reference", "target", "output", "ref")
_ID_HINTS = ("id", "row_id", "index", "no", "no.", "#")


def _fail(exc: Exception) -> typer.Exit:
    err.print(f"[bold red]Error:[/] {exc}")
    return typer.Exit(1)


@app.callback(invoke_without_command=True)
def _main(
    version: Annotated[bool, typer.Option("--version", help="Show the version and exit.")] = False,
) -> None:
    if version:
        console.print(f"paperlab-space {__version__}")
        raise typer.Exit()


@app.command()
def init(
    data: Annotated[Path, typer.Argument(help="Excel or CSV file to evaluate on.")],
    output: Annotated[Path, typer.Option("--output", "-o", help="Config file to write.")] = Path(
        "run.yaml"
    ),
    sheet: Annotated[str | None, typer.Option(help="Sheet name for Excel files.")] = None,
    force: Annotated[bool, typer.Option(help="Overwrite an existing config.")] = False,
) -> None:
    """Write a starter run config for DATA, guessing the column roles."""
    try:
        dataset = load_dataset(data, sheet)
    except PaperLabError as exc:
        raise _fail(exc) from exc
    if output.exists() and not force:
        raise _fail(FileExistsError(f"{output} exists; use --force to overwrite."))

    columns = dataset.columns
    lower = {c.lower(): c for c in columns}
    id_col = next((lower[h] for h in _ID_HINTS if h in lower), None)
    expected = next(
        (c for c in columns if any(h in c.lower() for h in _EXPECTED_HINTS) and c != id_col), None
    )
    inputs = [c for c in columns if c not in (id_col, expected)][:1] or columns[:1]

    try:
        relative = data.resolve().relative_to(output.resolve().parent)
    except ValueError:
        relative = data.resolve()
    config: dict[str, object] = {
        "dataset": {"path": str(relative), "sheet": dataset.sheet},
        "mapping": {"input": inputs, "context": [], "expected": expected, "id": id_col},
        "prompt": f"{{{inputs[0]}}}",
        "models": ["openrouter/nvidia/nemotron-3-super-120b-a12b:free", "gemini/gemini-2.5-flash"],
        "metrics": ["exact_match"] if expected else [],
        "seed": 42,
    }
    header = (
        f"# PaperLab run config for {data.name}\n"
        f"# Columns: {', '.join(columns)}\n"
        "# Edit the mapping, prompt, models, and metrics, then run:\n"
        f"#   paperlab-space estimate {output.name}\n"
        f"#   paperlab-space run {output.name}\n"
        f"# Metrics: {', '.join(sorted(available_metrics()))}\n"
        "# API keys are read from OPENROUTER_API_KEY, GEMINI_API_KEY, OPENAI_API_KEY, ...\n"
    )
    output.write_text(
        header + yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    console.print(f"Wrote [bold]{output}[/] (input: {inputs}, expected: {expected}, id: {id_col})")


@app.command()
def validate(
    data: Annotated[Path, typer.Argument(help="Excel or CSV file.")],
    sheet: Annotated[str | None, typer.Option(help="Sheet name for Excel files.")] = None,
    input: Annotated[
        list[str] | None, typer.Option("--input", "-i", help="Input column(s).")
    ] = None,
    expected: Annotated[str | None, typer.Option("--expected", "-e")] = None,
    id: Annotated[str | None, typer.Option("--id")] = None,
    delimiter: Annotated[str | None, typer.Option(help="Separator for multiple answers.")] = None,
    preview: Annotated[int, typer.Option(help="Rows to preview as JSONL.")] = 3,
) -> None:
    """Check a data file for problems and preview the converted JSONL."""
    try:
        dataset = load_dataset(data, sheet)
        console.print(
            f"[bold]{data.name}[/]: {len(dataset)} rows, sheet {dataset.sheet!r}"
            if dataset.sheet
            else f"[bold]{data.name}[/]: {len(dataset)} rows"
        )
        if data.suffix.lower() in (".xlsx", ".xlsm"):
            console.print(f"Sheets: {', '.join(list_sheets(data))}")
        console.print(f"Columns: {', '.join(dataset.columns)}")
        if not input:
            console.print("Pass --input (and --expected) to check a column mapping.")
            return
        mapping = ColumnMapping(input=input, expected=expected, id=id, answer_delimiter=delimiter)
        mapped = dataset.map(mapping)
    except (PaperLabError, ValueError) as exc:
        raise _fail(exc) from exc
    report = mapped.validate()
    console.print(str(report), markup=False, style="red" if not report.ok else None)
    if preview:
        console.print(f"\nFirst {min(preview, len(mapped))} records:")
        for line in mapped.to_jsonl().splitlines()[:preview]:
            console.print(line, markup=False)
    if not report.ok:
        raise typer.Exit(1)


@app.command("estimate")
def estimate_cmd(
    config: Annotated[Path, typer.Argument(help="Run config (YAML).")],
) -> None:
    """Estimate tokens, cost, and time for a run without calling any model."""
    try:
        cfg = RunConfig.from_yaml(config)
        console.print(str(estimate(cfg, prepare_data(cfg, None))), markup=False)
    except PaperLabError as exc:
        raise _fail(exc) from exc


@app.command("run")
def run_cmd(
    config: Annotated[Path, typer.Argument(help="Run config (YAML).")],
    out: Annotated[Path | None, typer.Option("--out", "-o", help="Output folder.")] = None,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip the confirmation.")] = False,
    resume: Annotated[
        Path | None, typer.Option(help="results.json of an interrupted run to continue.")
    ] = None,
) -> None:
    """Run an evaluation and write Excel, LaTeX, YAML, and JSON results."""
    out = out or config.parent / f"results-{config.stem}"
    try:
        cfg = RunConfig.from_yaml(config)
        data = prepare_data(cfg, None)
        previous = RunResults.load_json(resume) if resume else None
    except PaperLabError as exc:
        raise _fail(exc) from exc

    console.print(str(estimate(cfg, data)), markup=False)
    if not yes and not typer.confirm("Start the run?", default=True):
        raise typer.Exit()

    with Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("[red]{task.fields[failed]} failed"),
        TimeRemainingColumn(),
        console=console,
    ) as bar:
        tasks = {m.name: bar.add_task(m.name, total=len(data), failed=0) for m in cfg.models}

        def on_progress(p: RunProgress) -> None:
            bar.update(tasks[p.model], completed=p.done, failed=p.failed)

        try:
            results = run(cfg, data, on_progress=on_progress, previous=previous)
        except PaperLabError as exc:
            raise _fail(exc) from exc

    paths = write_all(results, out)
    _print_scores(results)
    failed = sum(r.n_failed for r in results.runs.values())
    if failed:
        console.print(
            f"[yellow]{failed} request(s) failed.[/] Re-run with "
            f"--resume {paths['json']} to retry only those rows."
        )
    console.print(
        f"Results written to [bold]{out}[/]: " + ", ".join(p.name for p in paths.values())
    )


@app.command()
def metrics() -> None:
    """List the available metrics."""
    table = Table("Metric", "Needs answer key", "Description")
    for name, cls in sorted(available_metrics().items()):
        table.add_row(name, "yes" if cls.requires_reference else "no", cls.description)
    console.print(table)


def _print_scores(results: RunResults) -> None:
    keys = [s.key for s in results.config.metrics]
    table = Table("Model", *keys, "Failed", "Cost (USD)")
    for name, model_run in results.runs.items():
        scores = results.scores.get(name, {})
        cells = [
            f"{scores[k].value:.4f}" if k in scores and scores[k].value is not None else "--"
            for k in keys
        ]
        cost = model_run.cost_usd
        table.add_row(
            name, *cells, str(model_run.n_failed), "--" if cost is None else f"{cost:.4f}"
        )
    console.print(table)
