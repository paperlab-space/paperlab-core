from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from paperlab_space.data import ColumnMapping, Dataset, load_dataset
from paperlab_space.errors import DataError


def test_map_and_jsonl(multilingual_xlsx: Path, tmp_path: Path) -> None:
    mapped = load_dataset(multilingual_xlsx).map(
        ColumnMapping(input="source", expected="gold", id="id")
    )
    assert len(mapped) == 5
    first = mapped.records[0]
    assert first.id == "1" and first.fields == {"source": "မင်္ဂလာပါ"}
    assert first.expected == ["hello"]
    out = tmp_path / "data.jsonl"
    text = mapped.to_jsonl(out)
    assert "မင်္ဂလာပါ" in text  # not \u-escaped
    lines = out.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[1]) == {"id": "2", "source": "สวัสดีครับ", "expected": ["hello"]}
    assert mapped.preview(2) == [json.loads(line) for line in lines[:2]]


def test_multiple_answers() -> None:
    ds = Dataset.from_records([{"q": "2+2", "a": "4 | four |"}])
    mapped = ds.map(ColumnMapping(input="q", expected="a", answer_delimiter="|"))
    assert mapped.records[0].expected == ["4", "four"]
    assert mapped.records[0].id == "1"


def test_mapping_rules() -> None:
    with pytest.raises(ValidationError, match="leak"):
        ColumnMapping(input=["q", "a"], expected="a")
    with pytest.raises(ValidationError, match="more than once"):
        ColumnMapping(input="q", context=["q"])
    with pytest.raises(ValidationError, match="At least one"):
        ColumnMapping(input=[])


def test_missing_column() -> None:
    ds = Dataset.from_records([{"q": "x"}])
    with pytest.raises(DataError, match="not found: answer"):
        ds.map(ColumnMapping(input="q", expected="answer"))


def test_validation_report() -> None:
    ds = Dataset.from_records(
        [
            {"id": "a", "q": "fine", "gold": "x"},
            {"id": "a", "q": None, "gold": "y", "note": "café"},
            {"id": None, "q": "cafÃ© ​", "gold": None},
        ]
    )
    report = ds.map(ColumnMapping(input="q", expected="gold", id="id")).validate()
    codes = {issue.code: issue for issue in report.issues}
    assert set(codes) == {
        "empty_input",
        "empty_expected",
        "empty_id",
        "duplicate_id",
        "encoding",
        "invisible_chars",
    }
    assert codes["duplicate_id"].rows == [1, 2]
    assert codes["empty_input"].level == "warning"
    assert not report.ok
    assert "error" in str(report)


def test_clean_report() -> None:
    ds = Dataset.from_records([{"q": "x", "a": "y"}])
    report = ds.map(ColumnMapping(input="q", expected="a")).validate()
    assert report.ok and report.issues == []
