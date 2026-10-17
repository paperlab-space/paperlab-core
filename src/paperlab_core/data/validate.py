"""Data quality checks run before an evaluation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from paperlab_core.data.mapping import MappedDataset

Level = Literal["error", "warning"]

# Characters that usually mean text was decoded with the wrong encoding.
_MOJIBAKE_MARKERS = ("�", "Ã©", "Ã¨", "Ã¼", "â€™", "â€œ", "â€\x9d", "Ã±")
_INVISIBLE = {"​", "﻿", " "}


@dataclass
class Issue:
    level: Level
    code: str
    message: str
    rows: list[int] = field(default_factory=list)
    """1-based data row numbers (header excluded)."""
    column: str | None = None

    def __str__(self) -> str:
        where = ""
        if self.rows:
            shown = ", ".join(map(str, self.rows[:10])) + (" ..." if len(self.rows) > 10 else "")
            where = f" (rows {shown})"
        return f"[{self.level}] {self.message}{where}"


@dataclass
class ValidationReport:
    issues: list[Issue]
    n_records: int

    @property
    def ok(self) -> bool:
        """``True`` if there are no errors (warnings are allowed)."""
        return not self.errors

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "warning"]

    def __str__(self) -> str:
        if not self.issues:
            return f"{self.n_records} records, no issues found."
        lines = [
            f"{self.n_records} records, {len(self.errors)} error(s), "
            f"{len(self.warnings)} warning(s):"
        ]
        lines += [f"  {issue}" for issue in self.issues]
        return "\n".join(lines)


def validate(mapped: MappedDataset) -> ValidationReport:
    mapping = mapped.mapping
    records = mapped.records
    issues: list[Issue] = []

    if not records:
        issues.append(Issue("error", "empty_dataset", "The dataset has no data rows."))
        return ValidationReport(issues, 0)

    for column in mapping.input:
        rows = [r.row_number for r in records if r.fields.get(column) is None]
        if rows:
            level: Level = "error" if len(rows) == len(records) else "warning"
            issues.append(
                Issue(
                    level, "empty_input", f"Input column {column!r} has empty cells", rows, column
                )
            )

    if mapping.expected is not None:
        rows = [r.row_number for r in records if r.expected is None]
        if rows:
            level = "error" if len(rows) == len(records) else "warning"
            issues.append(
                Issue(
                    level,
                    "empty_expected",
                    f"Expected column {mapping.expected!r} has empty cells; "
                    "these rows will not be scored",
                    rows,
                    mapping.expected,
                )
            )

    if mapping.id is not None:
        missing = [
            r.row_number
            for r in records
            if mapped.dataset.rows[r.row_number - 1].get(mapping.id) is None
        ]
        if missing:
            issues.append(
                Issue(
                    "error",
                    "empty_id",
                    f"ID column {mapping.id!r} has empty cells",
                    missing,
                    mapping.id,
                )
            )
        counts = Counter(r.id for r in records)
        dupes = [r.row_number for r in records if counts[r.id] > 1 and r.row_number not in missing]
        if dupes:
            issues.append(
                Issue(
                    "error",
                    "duplicate_id",
                    f"ID column {mapping.id!r} has duplicate values",
                    dupes,
                    mapping.id,
                )
            )

    for column in mapping.used_columns:
        values = [
            (r.row_number, mapped.dataset.rows[r.row_number - 1].get(column)) for r in records
        ]
        bad = [n for n, v in values if v is not None and any(m in v for m in _MOJIBAKE_MARKERS)]
        if bad:
            issues.append(
                Issue(
                    "warning",
                    "encoding",
                    f"Column {column!r} contains characters that suggest an encoding "
                    "problem (e.g. 'Ã©' or '�'); try re-saving the file as UTF-8",
                    bad,
                    column,
                )
            )
        hidden = [n for n, v in values if v is not None and any(ch in v for ch in _INVISIBLE)]
        if hidden:
            issues.append(
                Issue(
                    "warning",
                    "invisible_chars",
                    f"Column {column!r} contains invisible characters (zero-width "
                    "space, BOM, or non-breaking space) that can break exact matching",
                    hidden,
                    column,
                )
            )

    return ValidationReport(issues, len(records))
