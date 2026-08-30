from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

import fashion.train.artifacts as artifacts_module
from fashion.train.artifacts import (
    ArtifactVerificationError,
    atomic_write_bytes,
    atomic_write_csv,
    atomic_write_json,
    canonical_json_bytes,
    canonical_sha256,
    verify_artifact,
)


@dataclass(frozen=True)
class ExampleConfig:
    learning_rate: float
    output: Path


def test_canonical_hash_ignores_mapping_order_and_supports_configs() -> None:
    first = {"b": 2, "a": 1}
    second = {"a": 1, "b": 2}

    assert canonical_json_bytes(first) == b'{"a":1,"b":2}'
    assert canonical_sha256(first) == canonical_sha256(second)
    assert len(canonical_sha256(ExampleConfig(0.001, Path("runs/fold-0")))) == 64


def test_canonical_json_rejects_non_finite_numbers() -> None:
    with pytest.raises(ValueError, match="JSON compliant"):
        canonical_sha256({"loss": float("nan")})


def test_atomic_writes_replace_complete_artifacts(tmp_path: Path) -> None:
    binary_path = tmp_path / "nested" / "checkpoint.bin"
    json_path = tmp_path / "manifest.json"
    csv_path = tmp_path / "evidence.csv"

    atomic_write_bytes(binary_path, b"old")
    atomic_write_bytes(binary_path, b"new checkpoint")
    atomic_write_json(json_path, {"run_id": "task2-test", "fold": 0})
    atomic_write_csv(csv_path, pd.DataFrame({"id": [2, 1], "fold": [1, 0]}))

    assert binary_path.read_bytes() == b"new checkpoint"
    assert json.loads(json_path.read_text(encoding="utf-8"))["fold"] == 0
    assert csv_path.read_text(encoding="utf-8") == "id,fold\n2,1\n1,0\n"
    assert not list(tmp_path.rglob("*.tmp"))


def test_atomic_write_retries_transient_windows_replace_denial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "runs.csv"
    output.write_bytes(b"old")
    real_replace = artifacts_module.os.replace
    attempts = 0

    def deny_once(source: str | Path, destination: str | Path) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            error = PermissionError(5, "Access is denied", str(destination))
            error.winerror = 5
            raise error
        real_replace(source, destination)

    monkeypatch.setattr(artifacts_module.os, "replace", deny_once)

    atomic_write_bytes(output, b"new")

    assert attempts == 2
    assert output.read_bytes() == b"new"
    assert not list(tmp_path.glob("*.tmp"))


def test_verify_artifact_accepts_match_and_rejects_missing_or_changed(tmp_path: Path) -> None:
    artifact = tmp_path / "model.pt"
    atomic_write_bytes(artifact, b"scratch weights")
    expected = hashlib.sha256(b"scratch weights").hexdigest()

    assert verify_artifact(artifact, expected.upper()) == expected

    artifact.write_bytes(b"changed")
    with pytest.raises(ArtifactVerificationError, match="mismatch"):
        verify_artifact(artifact, expected)
    with pytest.raises(ArtifactVerificationError, match="does not exist"):
        verify_artifact(tmp_path / "missing.pt", expected)
