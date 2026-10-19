"""LaTeX export: a booktabs comparison table ready to ``\\input{}`` into a paper."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from paperlab_space.results import RunResults

METRIC_TITLES = {
    "exact_match": "EM",
    "accuracy": "Accuracy",
    "f1": "F1",
    "token_f1": "Token F1",
    "bleu": "BLEU",
    "chrf": "chrF",
    "rouge1": "ROUGE-1",
    "rouge2": "ROUGE-2",
    "rougeL": "ROUGE-L",
    "llm_judge": "Judge",
}

_SPECIAL = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def latex_table(
    results: RunResults,
    *,
    metrics: Sequence[str] | None = None,
    caption: str | None = None,
    label: str | None = "tab:results",
    decimals: int = 1,
    percent: bool = True,
    bold_best: bool = True,
    floating: bool = True,
    ci: bool = False,
    mark_significant: bool = True,
) -> str:
    """Build a model-by-metric comparison table.

    Args:
        metrics: Metric keys to include, in order. Defaults to all configured metrics.
        percent: Show scores in [0, 1] as percentages.
        bold_best: Bold the best score in each column (all of them on ties).
        floating: Wrap the tabular in a ``table`` environment with caption and label.
        ci: Show the confidence interval after each score, in brackets.
        mark_significant: With a ``statistics.baseline``, mark scores that differ
            significantly from the baseline's with a dagger.
    """
    keys = list(metrics) if metrics is not None else [s.key for s in results.config.metrics]
    unknown = [k for k in keys if k not in {s.key for s in results.config.metrics}]
    if unknown:
        raise ValueError(f"Unknown metric(s) for this run: {', '.join(unknown)}")

    values = {
        (model, key): results.scores.get(model, {}).get(key)
        for model in results.runs
        for key in keys
    }
    best: dict[str, float] = {}
    for key in keys:
        scored = [
            (r.value, r.higher_is_better)
            for (m, k), r in values.items()
            if k == key and r is not None and r.value is not None
        ]
        if scored:
            higher = scored[0][1]
            best[key] = max(v for v, _ in scored) if higher else min(v for v, _ in scored)

    settings = results.config.statistics
    significant: set[tuple[str, str]] = set()
    if mark_significant and settings.baseline is not None:
        significant = {
            (c.model_a, c.metric)
            for c in results.comparisons
            if c.significant and c.model_b == settings.baseline
        }

    def number(value: float) -> str:
        return f"{value * 100 if percent else value:.{decimals}f}"

    def cell(model: str, key: str) -> str:
        result = values[(model, key)]
        if result is None or result.value is None:
            return "--"
        text = number(result.value)
        if bold_best and len(results.runs) > 1 and round(result.value, 12) == round(best[key], 12):
            text = rf"\textbf{{{text}}}"
        if (model, key) in significant:
            text += r"$^\dagger$"
        if ci and result.ci_low is not None and result.ci_high is not None:
            text += rf" {{\scriptsize[{number(result.ci_low)}, {number(result.ci_high)}]}}"
        return text

    lines = []
    if floating:
        lines += [r"\begin{table}[t]", r"\centering"]
    lines += [rf"\begin{{tabular}}{{l{'r' * len(keys)}}}", r"\toprule"]
    lines.append(" & ".join(["Model", *[escape(METRIC_TITLES.get(k, k)) for k in keys]]) + r" \\")
    lines.append(r"\midrule")
    for model in results.runs:
        lines.append(" & ".join([escape(model), *[cell(model, k) for k in keys]]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if floating:
        if caption is None:
            n = len(results.data.records)
            unit = " (\\%)" if percent else ""
            caption = f"Results on {n} examples{unit}."
        else:
            caption = escape(caption)
        shown_ci = ci and any(r is not None and r.ci_low is not None for r in values.values())
        if shown_ci:
            caption += (
                f" Brackets: {settings.confidence:.0%} bootstrap confidence intervals.".replace(
                    "%", r"\%"
                )
            )
        if significant & {(m, k) for m in results.runs for k in keys}:
            test = (
                "paired permutation"
                if settings.significance == "permutation"
                else ("paired bootstrap")
            )
            correction = (
                f", {settings.correction.capitalize()}-corrected"
                if settings.correction != "none"
                else ""
            )
            caption += (
                rf" $^\dagger$: significantly different from {escape(settings.baseline or '')}"
                rf" ({test} test{correction}, $p < {1 - settings.confidence:.2g}$)."
            )
        lines.append(rf"\caption{{{caption}}}")
        if label:
            lines.append(rf"\label{{{label}}}")
        lines.append(r"\end{table}")

    version = results.environment.get("paperlab-space", "unknown")
    header = [f"% Generated by paperlab-space {version}", r"% Requires \usepackage{booktabs}"]
    return "\n".join(header + lines) + "\n"


def write_latex(results: RunResults, path: str | Path, **options: Any) -> Path:
    """Write :func:`latex_table` output to ``path``. Options are passed through."""
    path = Path(path)
    path.write_text(latex_table(results, **options), encoding="utf-8")
    return path


def escape(text: str) -> str:
    """Escape LaTeX special characters."""
    return "".join(_SPECIAL.get(ch, ch) for ch in text)
