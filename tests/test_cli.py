from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from paperlab_space._cli import app
from paperlab_space.config import ModelSpec
from paperlab_space.runner import runner as runner_module

from .fakes import FakeBackend, ProviderError

cli = CliRunner()


def _answer(model: ModelSpec, prompt: str, attempt: int) -> str | Exception:
    if "你好" in prompt and attempt == 1 and model.id == "openai/b":
        return ProviderError("temporary")
    return "hello"


@pytest.fixture
def fake_backend(monkeypatch: pytest.MonkeyPatch) -> FakeBackend:
    backend = FakeBackend(_answer)
    monkeypatch.setattr(runner_module, "LiteLLMBackend", lambda: backend)
    return backend


def test_init_validate_estimate_run_resume(
    multilingual_xlsx: Path, tmp_path: Path, fake_backend: FakeBackend
) -> None:
    config = tmp_path / "run.yaml"
    result = cli.invoke(app, ["init", str(multilingual_xlsx), "-o", str(config)])
    assert result.exit_code == 0, result.output
    data = yaml.safe_load(config.read_text(encoding="utf-8"))
    assert data["mapping"] == {"input": ["source"], "context": [], "expected": "gold", "id": "id"}
    assert data["dataset"]["path"] == "data.xlsx"
    assert cli.invoke(app, ["init", str(multilingual_xlsx), "-o", str(config)]).exit_code == 1
    # Use unthrottled models so the test does not wait on free-tier rate limits.
    data["models"] = ["openai/a", "openai/b"]
    config.write_text(yaml.safe_dump(data), encoding="utf-8")

    result = cli.invoke(
        app, ["validate", str(multilingual_xlsx), "-i", "source", "-e", "gold", "--id", "id"]
    )
    assert result.exit_code == 0, result.output
    assert "no issues" in result.output and "မင်္ဂလာပါ" in result.output

    result = cli.invoke(app, ["estimate", str(config)])
    assert result.exit_code == 0, result.output
    assert "Total" in result.output

    out = tmp_path / "out"
    result = cli.invoke(app, ["run", str(config), "-o", str(out), "--yes"])
    assert result.exit_code == 0, result.output
    assert "1 request(s) failed" in result.output
    assert (out / "results.xlsx").exists() and (out / "table.tex").exists()

    calls_before = len(fake_backend.calls)
    result = cli.invoke(
        app, ["run", str(config), "-o", str(out), "--yes", "--resume", str(out / "results.json")]
    )
    assert result.exit_code == 0, result.output
    assert len(fake_backend.calls) == calls_before + 1  # only the failed row is retried
    assert "failed." not in result.output


def test_validate_reports_errors(tmp_path: Path) -> None:
    path = tmp_path / "d.csv"
    path.write_text("id,q\n1,a\n1,b\n", encoding="utf-8")
    result = cli.invoke(app, ["validate", str(path), "-i", "q", "--id", "id"])
    assert result.exit_code == 1
    assert "duplicate" in result.output
    result = cli.invoke(app, ["validate", str(path), "-i", "missing"])
    assert result.exit_code == 1 and "not found" in result.output


def test_metrics_and_version() -> None:
    result = cli.invoke(app, ["metrics"])
    assert result.exit_code == 0 and "token_f1" in result.output
    result = cli.invoke(app, ["--version"])
    assert result.exit_code == 0 and "paperlab-space" in result.output
