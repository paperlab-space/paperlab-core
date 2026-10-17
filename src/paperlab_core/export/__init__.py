"""Writing results: Excel, LaTeX, reproducibility config, and JSON."""

from __future__ import annotations

from pathlib import Path

from paperlab_core.export.excel import write_excel
from paperlab_core.export.latex import latex_table, write_latex
from paperlab_core.export.repro import reproducibility, write_repro
from paperlab_core.results import RunResults


def write_all(results: RunResults, out_dir: str | Path) -> dict[str, Path]:
    """Write every export format into ``out_dir`` and return the paths by format."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "results.json"
    results.save_json(json_path)
    return {
        "excel": write_excel(results, out / "results.xlsx"),
        "latex": write_latex(results, out / "table.tex"),
        "repro": write_repro(results, out / "run.yaml"),
        "json": json_path,
    }


__all__ = [
    "latex_table",
    "reproducibility",
    "write_all",
    "write_excel",
    "write_latex",
    "write_repro",
]
