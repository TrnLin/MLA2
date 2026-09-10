from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from types import ModuleType
from typing import Any

import pytest

from fashion.config import ROOT

CLI = ROOT / "scripts/build_task4_final_evaluation.py"
PYTHON = ROOT / ".venv/bin/python"


def _load_cli() -> ModuleType:
    spec = importlib.util.spec_from_file_location("task4_final_evaluation_cli", CLI)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Task 4 final-evaluation CLI")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parser_requires_a_subcommand() -> None:
    cli = _load_cli()

    with pytest.raises(SystemExit) as caught:
        cli._parser().parse_args([])

    assert caught.value.code == 2


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["score"], False),
        (["score", "--evaluation-unlocked"], True),
    ],
    ids=["locked-by-default", "explicitly-unlocked"],
)
def test_score_parser_requires_explicit_unlock_flag(
    argv: list[str],
    expected: bool,
) -> None:
    cli = _load_cli()

    args = cli._parser().parse_args(argv)

    assert args.command == "score"
    assert args.evaluation_unlocked is expected


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["predict"], "cpu"),
        (["predict", "--device", "cuda"], "cuda"),
    ],
    ids=["cpu-default", "cuda-explicit"],
)
def test_predict_parser_selects_supported_device(
    argv: list[str],
    expected: str,
) -> None:
    cli = _load_cli()

    args = cli._parser().parse_args(argv)

    assert args.command == "predict"
    assert args.device == expected


@pytest.mark.parametrize(
    ("argv", "function_name", "expected_kwargs"),
    [
        (["predict", "--device", "cuda"], "build_blind_holdout_evidence", {"device": "cuda"}),
        (
            ["score", "--evaluation-unlocked"],
            "score_holdout",
            {"evaluation_unlocked": True},
        ),
        (["audit"], "load_verified_holdout_evaluation", {}),
    ],
    ids=["predict", "score", "audit"],
)
def test_main_dispatches_once_and_prints_only_safe_summary(
    argv: list[str],
    function_name: str,
    expected_kwargs: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _load_cli()
    calls: list[tuple[str, dict[str, object]]] = []
    result = {
        "evaluation_id": "task4-final",
        "phase": "blind",
        "status": "complete",
        "labels_opened": False,
        "coverage": {"queries": 10},
        "artifact_path": "/protected/evaluation.json",
        "rankings": [{"query_id": 1, "candidate_ids": [2, 3]}],
    }

    def fake_dispatch(**kwargs: object) -> dict[str, Any]:
        calls.append((function_name, kwargs))
        return result

    def fail_dispatch(**kwargs: object) -> dict[str, Any]:
        raise AssertionError(f"unexpected dispatch with {kwargs}")

    for name in (
        "build_blind_holdout_evidence",
        "score_holdout",
        "load_verified_holdout_evaluation",
    ):
        monkeypatch.setattr(cli, name, fake_dispatch if name == function_name else fail_dispatch)
    monkeypatch.setattr(sys, "argv", [str(CLI), *argv])

    assert cli.main() == 0

    assert calls == [(function_name, expected_kwargs)]
    assert json.loads(capsys.readouterr().out) == {
        "coverage": {"queries": 10},
        "evaluation_id": "task4-final",
        "labels_opened": False,
        "phase": "blind",
        "status": "complete",
    }


def test_score_without_flag_forwards_false(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _load_cli()
    received: list[bool] = []

    def fake_score_holdout(*, evaluation_unlocked: bool) -> dict[str, object]:
        received.append(evaluation_unlocked)
        return {"status": "refused"}

    monkeypatch.setattr(cli, "score_holdout", fake_score_holdout)
    monkeypatch.setattr(sys, "argv", [str(CLI), "score"])

    assert cli.main() == 0

    assert received == [False]
    assert json.loads(capsys.readouterr().out) == {"status": "refused"}


def test_help_lists_all_verbs_and_exits_zero() -> None:
    completed = subprocess.run(
        [str(PYTHON), str(CLI), "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    assert all(verb in completed.stdout for verb in ("predict", "score", "audit"))
