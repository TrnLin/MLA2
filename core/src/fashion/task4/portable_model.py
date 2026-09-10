"""Portable inference-only package for the selected Task 4 R5 model."""

from __future__ import annotations

import hashlib
import json
import math
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from fashion.task4.models import EMBEDDING_DIM, SCRATCH_WEIGHT_ORIGIN, build_autoencoder

PORTABLE_R5_SCHEMA_VERSION = "1.0.0"
_WEIGHTS_FILENAME = "weights.pt"
_MANIFEST_FILENAME = "manifest.json"
_README_FILENAME = "README.md"
_PACKAGE_FILES = (_README_FILENAME, _MANIFEST_FILENAME, _WEIGHTS_FILENAME)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} cannot be read: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return payload


def _source_payload(path: Path) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError(f"source checkpoint cannot be loaded: {error}") from error
    if not isinstance(payload, Mapping):
        raise ValueError("source checkpoint must contain one mapping")

    try:
        config = json.loads(payload["canonical_config_json"])
        candidate = config["candidate"]
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("source checkpoint configuration is invalid") from error
    if candidate != {
        "architecture": "resnet18",
        "candidate": "R5",
        "pretrained": False,
    }:
        raise ValueError("source checkpoint is not the scratch R5 model")
    if (
        config.get("objective") != "content_mask_mse"
        or payload.get("weight_origin") != SCRATCH_WEIGHT_ORIGIN
    ):
        raise ValueError("source checkpoint has the wrong R5 training identity")
    return payload


def _validated_model_state(payload: Mapping[str, Any]) -> tuple[nn.Module, dict[str, torch.Tensor]]:
    raw_state = payload.get("model_state_dict")
    if not isinstance(raw_state, Mapping) or not raw_state:
        raise ValueError("source checkpoint model state is missing")
    if any(
        not isinstance(name, str) or not isinstance(value, torch.Tensor)
        for name, value in raw_state.items()
    ):
        raise ValueError("source checkpoint model state is invalid")

    model = build_autoencoder()
    state = {name: value.detach().cpu() for name, value in raw_state.items()}
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise ValueError(f"source checkpoint does not fit the R5 model: {error}") from error
    model.eval()
    return model, state


def _input_contract(path: Path) -> dict[str, Any]:
    payload = _read_json(path, label="preprocessing contract")
    value = payload.get("input_contract")
    expected = {
        "colour_mode": "RGB",
        "height": 320,
        "pad_color": [255, 255, 255],
        "resample": "LANCZOS",
        "resize": "aspect_preserving_letterbox",
        "width": 240,
    }
    if value != expected:
        raise ValueError("preprocessing contract does not match the selected Task 4 input")
    return expected


def _normalization(path: Path) -> dict[str, dict[str, list[float]]]:
    payload = _read_json(path, label="normalization artifact")
    sources = payload.get("sources")
    if not isinstance(sources, Mapping) or set(sources) != {"teacher", "v1"}:
        raise ValueError("normalization artifact must contain teacher and v1 statistics")
    result: dict[str, dict[str, list[float]]] = {}
    for source in ("teacher", "v1"):
        value = sources[source]
        if not isinstance(value, Mapping):
            raise ValueError(f"{source} normalization statistics are invalid")
        result[source] = {}
        for field in ("mean", "std"):
            raw = value.get(field)
            if (
                not isinstance(raw, list)
                or len(raw) != 3
                or any(
                    isinstance(item, bool)
                    or not isinstance(item, (int, float))
                    or not math.isfinite(float(item))
                    or (field == "std" and float(item) <= 0.0)
                    for item in raw
                )
            ):
                raise ValueError(f"{source} normalization {field} is invalid")
            result[source][field] = [float(item) for item in raw]
    return result


def _readme() -> str:
    return """# Task 4 R5 portable model

This folder contains inference-only weights for the selected scratch R5 visual-search model.
It does not contain optimizer, scheduler, gradient-scaler, training images, or index data.

## Load

Install the MLA2 project and its pinned packages, then run:

```python
from fashion.task4 import load_r5_inference_package

model = load_r5_inference_package("task4_r5", device="cpu")
```

Read `manifest.json` for the required 240x320 RGB letterbox input and the teacher/V1
normalization values. Call `model.encode(batch)` while the model is in evaluation mode to
produce normalized 128-value retrieval embeddings.
"""


def export_r5_inference_package(
    source_checkpoint: str | Path,
    destination_directory: str | Path,
    *,
    preprocessing_contract_path: str | Path,
    normalization_path: str | Path,
    archive_path: str | Path,
    expected_source_sha256: str | None = None,
    expected_run_id: str | None = None,
) -> tuple[Path, Path]:
    """Export validated R5 weights, portable metadata, instructions, and a ZIP archive."""

    source = Path(source_checkpoint)
    destination = Path(destination_directory)
    archive = Path(archive_path)
    if destination.exists():
        raise ValueError(f"destination already exists: {destination}")
    if archive.exists():
        raise ValueError(f"archive already exists: {archive}")

    source_sha256 = _sha256_file(source)
    if expected_source_sha256 is not None and source_sha256 != expected_source_sha256:
        raise ValueError("source checkpoint SHA-256 does not match the selected R5 checkpoint")
    payload = _source_payload(source)
    run_id = payload.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("source checkpoint run ID is invalid")
    if expected_run_id is not None and run_id != expected_run_id:
        raise ValueError("source checkpoint run ID does not match the selected R5 run")
    _, state = _validated_model_state(payload)
    contract = _input_contract(Path(preprocessing_contract_path))
    normalization = _normalization(Path(normalization_path))

    destination.mkdir(parents=True)
    weights_path = destination / _WEIGHTS_FILENAME
    torch.save(
        {
            "schema_version": PORTABLE_R5_SCHEMA_VERSION,
            "model_state_dict": state,
        },
        weights_path,
    )
    manifest = {
        "schema_version": PORTABLE_R5_SCHEMA_VERSION,
        "artifact_type": "task4_r5_inference_package",
        "method": "R5",
        "architecture": "resnet18",
        "objective": "content_mask_mse",
        "pretrained": False,
        "weight_origin": SCRATCH_WEIGHT_ORIGIN,
        "embedding_dim": EMBEDDING_DIM,
        "source_checkpoint": {
            "run_id": run_id,
            "epoch": int(payload["epoch"]),
            "score": float(payload["score"]),
            "sha256": source_sha256,
        },
        "weights": {
            "path": _WEIGHTS_FILENAME,
            "sha256": _sha256_file(weights_path),
            "bytes": weights_path.stat().st_size,
        },
        "input_contract": contract,
        "normalization": normalization,
        "loader": {
            "module": "fashion.task4",
            "function": "load_r5_inference_package",
        },
    }
    (destination / _MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (destination / _README_FILENAME).write_text(_readme(), encoding="utf-8")

    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary_archive = archive.with_suffix(f"{archive.suffix}.tmp")
    try:
        with zipfile.ZipFile(
            temporary_archive,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as bundle:
            for filename in _PACKAGE_FILES:
                bundle.write(destination / filename, f"{destination.name}/{filename}")
        temporary_archive.replace(archive)
    finally:
        temporary_archive.unlink(missing_ok=True)
    return destination, archive


def load_r5_inference_package(
    package_directory: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> nn.Module:
    """Validate and load a portable R5 package for embedding inference."""

    package = Path(package_directory)
    manifest = _read_json(package / _MANIFEST_FILENAME, label="portable R5 manifest")
    if (
        manifest.get("schema_version") != PORTABLE_R5_SCHEMA_VERSION
        or manifest.get("artifact_type") != "task4_r5_inference_package"
        or manifest.get("method") != "R5"
        or manifest.get("architecture") != "resnet18"
        or manifest.get("pretrained") is not False
        or manifest.get("weight_origin") != SCRATCH_WEIGHT_ORIGIN
        or manifest.get("embedding_dim") != EMBEDDING_DIM
    ):
        raise ValueError("portable R5 manifest identity is invalid")
    weights_record = manifest.get("weights")
    if not isinstance(weights_record, Mapping):
        raise ValueError("portable R5 weight record is invalid")
    relative = Path(str(weights_record.get("path") or ""))
    if relative != Path(_WEIGHTS_FILENAME):
        raise ValueError("portable R5 weight path is invalid")
    weights_path = package / relative
    expected_sha256 = weights_record.get("sha256")
    if not isinstance(expected_sha256, str) or _sha256_file(weights_path) != expected_sha256:
        raise ValueError("portable R5 weights SHA-256 does not match")

    try:
        payload = torch.load(weights_path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError(f"portable R5 weights cannot be loaded: {error}") from error
    if not isinstance(payload, Mapping) or set(payload) != {
        "model_state_dict",
        "schema_version",
    }:
        raise ValueError("portable R5 weight payload is invalid")
    if payload["schema_version"] != PORTABLE_R5_SCHEMA_VERSION:
        raise ValueError("portable R5 weight schema does not match")
    model = build_autoencoder()
    try:
        model.load_state_dict(payload["model_state_dict"], strict=True)
    except (TypeError, RuntimeError) as error:
        raise ValueError(f"portable R5 model state is invalid: {error}") from error
    return model.to(device).eval()
