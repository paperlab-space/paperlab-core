"""Map spreadsheet columns to the roles an evaluation needs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from paperlab_space.data.load import Dataset
from paperlab_space.errors import DataError

if TYPE_CHECKING:
    from paperlab_space.data.validate import ValidationReport


class ColumnMapping(BaseModel):
    """Which columns are inputs, extra context, the expected answer, and the row ID.

    Input and context columns become variables in the prompt template. The expected
    column is never exposed to the prompt, so answers cannot leak into it.
    """

    model_config = ConfigDict(extra="forbid")

    input: list[str]
    context: list[str] = []
    expected: str | None = None
    id: str | None = None
    answer_delimiter: str | None = None
    """Split the expected cell on this string to allow several valid answers, e.g. ``"|"``."""

    @field_validator("input", "context", mode="before")
    @classmethod
    def _one_or_many(cls, value: Any) -> Any:
        return [value] if isinstance(value, str) else value

    @model_validator(mode="after")
    def _check_roles(self) -> ColumnMapping:
        if not self.input:
            raise ValueError("At least one input column is required.")
        fields = self.input + self.context
        duplicates = sorted({name for name in fields if fields.count(name) > 1})
        if duplicates:
            raise ValueError(f"Columns used more than once: {', '.join(duplicates)}")
        if self.expected is not None and self.expected in fields:
            raise ValueError(
                f"Expected-output column {self.expected!r} cannot also be an input or context "
                "column, or the answer would leak into the prompt."
            )
        if self.answer_delimiter == "":
            raise ValueError("answer_delimiter cannot be empty.")
        return self

    @property
    def fields(self) -> list[str]:
        """Columns available to the prompt template."""
        return self.input + self.context

    @property
    def used_columns(self) -> list[str]:
        extra = [c for c in (self.expected, self.id) if c is not None and c not in self.fields]
        return self.fields + list(dict.fromkeys(extra))


@dataclass
class Record:
    """One evaluation example."""

    id: str
    fields: dict[str, str | None]
    expected: list[str] | None
    row_number: int
    """1-based position among the data rows (header excluded)."""

    def to_json(self) -> dict[str, Any]:
        data: dict[str, Any] = {"id": self.id, **self.fields}
        if self.expected is not None:
            data["expected"] = self.expected
        return data


@dataclass
class MappedDataset:
    """A dataset with a column mapping applied, ready for prompting and scoring."""

    records: list[Record]
    mapping: ColumnMapping
    dataset: Dataset

    @classmethod
    def from_dataset(cls, dataset: Dataset, mapping: ColumnMapping) -> MappedDataset:
        missing = [c for c in mapping.used_columns if c not in dataset.columns]
        if missing:
            raise DataError(
                f"Column(s) not found: {', '.join(missing)}. "
                f"Available columns: {', '.join(dataset.columns)}"
            )
        records = []
        for number, row in enumerate(dataset.rows, start=1):
            record_id = row.get(mapping.id) if mapping.id else None
            records.append(
                Record(
                    id=record_id if record_id is not None else str(number),
                    fields={name: row.get(name) for name in mapping.fields},
                    expected=_split_answers(row.get(mapping.expected), mapping.answer_delimiter)
                    if mapping.expected
                    else None,
                    row_number=number,
                )
            )
        return cls(records=records, mapping=mapping, dataset=dataset)

    def __len__(self) -> int:
        return len(self.records)

    @property
    def has_answer_key(self) -> bool:
        return self.mapping.expected is not None

    def validate(self) -> ValidationReport:
        """Check the data for empty cells, duplicate IDs, and encoding problems."""
        from paperlab_space.data.validate import validate

        return validate(self)

    def preview(self, n: int = 5) -> list[dict[str, Any]]:
        """The first ``n`` records as JSON-ready dicts."""
        return [record.to_json() for record in self.records[:n]]

    def to_jsonl(self, path: str | Path | None = None) -> str:
        """Serialise all records as JSON Lines (UTF-8, non-ASCII kept as-is)."""
        text = "".join(
            json.dumps(record.to_json(), ensure_ascii=False) + "\n" for record in self.records
        )
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text


def _split_answers(value: str | None, delimiter: str | None) -> list[str] | None:
    if value is None:
        return None
    if delimiter is None:
        return [value]
    answers = [part.strip() for part in value.split(delimiter)]
    return [a for a in answers if a] or None
