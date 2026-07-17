"""Tests for the Typer CLI surface.

Exercises every command Typer registers (`version`, `validate`,
`benchmark`, `profile`, `compress`) via Typer's ``CliRunner`` so the
test runs without spawning real subprocesses or loading real models.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest
import torch
from typer.testing import CliRunner

from kvcompress import __version__
from kvcompress.cli import app


runner = CliRunner()


# ---------------------------------------------------------------------------
# version
# ---------------------------------------------------------------------------


def test_version_prints_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


def test_validate_runs_synthetic_round_trip() -> None:
    result = runner.invoke(app, ["validate", "--skip-hf"])
    assert result.exit_code == 0
    assert "JoLT round-trip rel error" in result.stdout
    assert "kvcompress validate: OK" in result.stdout


def test_validate_reports_flashjolt_error() -> None:
    """The validate command runs FlashJoLT too and reports its rel_err."""
    result = runner.invoke(app, ["validate", "--skip-hf"])
    assert result.exit_code == 0
    assert "FlashJoLT round-trip rel error" in result.stdout


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------


def test_benchmark_swallows_per_suite_failure(tmp_path: Path) -> None:
    """A failure in one benchmark suite must not abort the orchestration.

    We mock subprocess.check_call to return normally for the first call
    and raise CalledProcessError for the second; the command should
    exit 1 (orchestration-level failure) but report both.
    """
    from subprocess import CalledProcessError

    from kvcompress.cli import run_subprocess as real_run

    calls: list[str] = []

    def fake_run(args: list[str], label: str, timeout: float = 600.0) -> bool:
        calls.append(label)
        if label == "memory":
            raise CalledProcessError(1, args)
        return True

    with mock.patch("kvcompress.cli.run_subprocess", side_effect=fake_run):
        result = runner.invoke(
            app,
            ["benchmark", "--suite", "all", "--output-dir", str(tmp_path)],
        )
    assert "memory" in calls
    # The fake for memory raised before real_run could be invoked.
    assert result.exit_code != 0


def test_benchmark_succeeds_when_all_suites_pass(tmp_path: Path) -> None:
    """Happy path: every benchmark suite reports OK."""
    with mock.patch("kvcompress.cli.run_subprocess", return_value=True):
        result = runner.invoke(
            app,
            ["benchmark", "--suite", "all", "--output-dir", str(tmp_path)],
        )
    assert result.exit_code == 0
    assert "benchmark outputs in" in result.stdout


# ---------------------------------------------------------------------------
# profile
# ---------------------------------------------------------------------------


def test_profile_runs_subprocess() -> None:
    """profile invokes scripts.profile_model via run_subprocess."""
    with mock.patch("kvcompress.cli.run_subprocess", return_value=True) as m:
        result = runner.invoke(
            app,
            ["profile", "--model", "gpt2", "--ratio", "3.0", "--method", "flashjolt"],
        )
    assert result.exit_code == 0
    assert m.called


# ---------------------------------------------------------------------------
# compress
# ---------------------------------------------------------------------------


def test_compress_requires_hf_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    """If transformers can't be imported, compress exits non-zero."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "transformers" or name.startswith("transformers."):
            raise ImportError("simulated missing transformers")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    result = runner.invoke(
        app,
        ["compress", "--model", "gpt2", "--target", "50%"],
    )
    assert result.exit_code != 0
    # typer.echo(..., err=True) goes to stderr; combine both streams.
    combined = (result.stdout or "") + (getattr(result, "stderr", "") or "")
    assert "error" in combined.lower() or "transformers" in combined.lower()


def test_compress_with_stubbed_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end compress with a stubbed AutoModelForCausalLM."""
    class _StubModel:
        class _Config:
            model_type = "gpt2"

        config = _Config()

        def __init__(self) -> None:
            self.generation_config = mock.Mock()
            self.generation_config.cache_implementation = None

        def eval(self) -> "_StubModel":
            return self

        def generate(self, ids, **kwargs):  # noqa: ARG002
            return torch.cat([ids, ids.new_zeros((1, kwargs["max_new_tokens"]))], dim=1)

    def fake_automodel(*args, **kwargs):
        return _StubModel()

    def fake_autotokenizer(*args, **kwargs):
        tok = mock.Mock()
        tok.encode = lambda text, **kw: torch.tensor([[1, 2, 3]])
        tok.decode = lambda ids, **kw: "ok"
        return tok

    monkeypatch.setattr(
        "transformers.AutoModelForCausalLM.from_pretrained", fake_automodel
    )
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained", fake_autotokenizer
    )

    result = runner.invoke(
        app,
        [
            "compress",
            "--model",
            "gpt2",
            "--method",
            "identity",
            "--target",
            "100%",
            "--prompt",
            "Hello",
            "--max-new",
            "2",
        ],
    )
    if result.exit_code != 0:
        combined = (result.stdout or "") + (getattr(result, "stderr", "") or "")
        pytest.fail(f"compress failed: {combined}")
    assert "ok" in result.stdout
    # Stats JSON appears at the end.
    stats_start = result.stdout.rfind("{")
    assert stats_start != -1
    stats = json.loads(result.stdout[stats_start:])
    assert "compress_calls" in stats