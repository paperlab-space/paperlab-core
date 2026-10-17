from __future__ import annotations

import datetime as dt
import unicodedata
from collections.abc import Callable
from pathlib import Path

import pytest

from paperlab_space.data import Dataset, list_sheets, load_dataset
from paperlab_space.errors import DataError

from .conftest import MULTILINGUAL


def test_load_xlsx_multilingual(multilingual_xlsx: Path) -> None:
    ds = load_dataset(multilingual_xlsx)
    assert ds.columns == ["id", "source", "gold"]
    assert ds.sheet == "test"
    assert [row["source"] for row in ds.rows] == [m[1] for m in MULTILINGUAL]
    assert ds.sha256 is not None and len(ds.sha256) == 64


def test_select_sheet_by_name_and_index(make_xlsx: Callable[..., Path]) -> None:
    path = make_xlsx({"a": [("x",), ("1",)], "b": [("y",), ("2",)]})
    assert list_sheets(path) == ["a", "b"]
    assert load_dataset(path, sheet="b").columns == ["y"]
    assert load_dataset(path, sheet=1).rows == [{"y": "2"}]
    with pytest.raises(DataError, match="not found"):
        load_dataset(path, sheet="c")
    with pytest.raises(DataError, match="out of range"):
        load_dataset(path, sheet=5)


def test_excel_cell_types_become_text(make_xlsx: Callable[..., Path]) -> None:
    path = make_xlsx(
        {
            "s": [
                ("int", "float", "whole", "bool", "date", "empty"),
                (3, 0.25, 2.0, True, dt.datetime(2026, 1, 2), "   "),
            ]
        }
    )
    row = load_dataset(path).rows[0]
    assert row == {
        "int": "3",
        "float": "0.25",
        "whole": "2",
        "bool": "TRUE",
        "date": "2026-01-02",
        "empty": None,
    }


def test_messy_header_and_empty_rows(make_xlsx: Callable[..., Path]) -> None:
    path = make_xlsx(
        {
            "s": [
                (None, None),
                ("name", None, "name", "  spaced   out "),
                ("a", "b", "c", "d"),
                (None, None, None, None),
                ("e", None, None, None),
            ]
        }
    )
    ds = load_dataset(path)
    assert ds.columns == ["name", "column_2", "name_2", "spaced out"]
    assert len(ds) == 2
    assert ds.rows[1] == {"name": "e", "column_2": None, "name_2": None, "spaced out": None}


def test_csv_utf8_bom_and_semicolon(tmp_path: Path) -> None:
    path = tmp_path / "d.csv"
    path.write_bytes("﻿id;text\n1;สวัสดี\n".encode())
    ds = load_dataset(path)
    assert ds.columns == ["id", "text"]
    assert ds.rows == [{"id": "1", "text": "สวัสดี"}]
    assert ds.encoding == "utf-8-sig"


@pytest.mark.parametrize(
    ("encoding", "text"),
    [
        ("cp874", "ภาษาไทยเป็นภาษาที่สวยงามมาก"),
        ("shift_jis", "日本語のテキストです。これはテストです。"),
        ("gbk", "这是一个中文测试句子，用于检测编码。"),
    ],
)
def test_csv_legacy_encodings(tmp_path: Path, encoding: str, text: str) -> None:
    path = tmp_path / "d.csv"
    path.write_bytes(f"text\n{text}\n{text}\n".encode(encoding))
    assert load_dataset(path).rows[0]["text"] == text


def test_tsv(tmp_path: Path) -> None:
    path = tmp_path / "d.tsv"
    path.write_text("a\tb\n1, 2\t3\n", encoding="utf-8")
    assert load_dataset(path).rows == [{"a": "1, 2", "b": "3"}]


def test_nfc_normalisation(tmp_path: Path) -> None:
    decomposed = unicodedata.normalize("NFD", "한국어 café")
    path = tmp_path / "d.csv"
    path.write_text(f"t\n{decomposed}\n", encoding="utf-8")
    assert load_dataset(path).rows[0]["t"] == unicodedata.normalize("NFC", decomposed)


def test_errors(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="not found"):
        load_dataset(tmp_path / "missing.csv")
    (tmp_path / "old.xls").write_bytes(b"x")
    with pytest.raises(DataError, match="re-save"):
        load_dataset(tmp_path / "old.xls")
    (tmp_path / "empty.csv").write_text("", encoding="utf-8")
    with pytest.raises(DataError, match="empty"):
        load_dataset(tmp_path / "empty.csv")


def test_from_records() -> None:
    ds = Dataset.from_records([{"a": 1, "b": "x"}, {"a": 2, "c": None}, {}])
    assert ds.columns == ["a", "b", "c"]
    assert ds.rows == [{"a": "1", "b": "x", "c": None}, {"a": "2", "b": None, "c": None}]
