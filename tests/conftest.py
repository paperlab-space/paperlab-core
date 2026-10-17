from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest
from openpyxl import Workbook

MULTILINGUAL = [
    ("1", "မင်္ဂလာပါ", "hello"),  # Burmese
    ("2", "สวัสดีครับ", "hello"),  # Thai
    ("3", "你好", "hello"),  # Chinese
    ("4", "ありがとう", "thank you"),  # Japanese
    ("5", "감사합니다", "thank you"),  # Korean
]


@pytest.fixture
def make_xlsx(tmp_path: Path) -> Callable[..., Path]:
    def make(sheets: dict[str, Sequence[Sequence[Any]]], name: str = "data.xlsx") -> Path:
        workbook = Workbook()
        workbook.remove(workbook.active)  # type: ignore[arg-type]
        for title, rows in sheets.items():
            sheet = workbook.create_sheet(title)
            for row in rows:
                sheet.append(list(row))
        path = tmp_path / name
        workbook.save(path)
        return path

    return make


@pytest.fixture
def multilingual_xlsx(make_xlsx: Callable[..., Path]) -> Path:
    return make_xlsx({"test": [("id", "source", "gold"), *MULTILINGUAL]})
