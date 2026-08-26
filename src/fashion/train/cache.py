"""Hash-validated reuse of completed experiment runs."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from fashion.config import LABEL_MAPS_JSON, ROOT, SPLITS_CSV
from fashion.data.hashing import compute_sha256
from fashion.train.artifacts import ArtifactVerificationError, canonical_sha256, verify_artifact
from fashion.train.registry import RunRegistry

ARTIFACT_FIELD_PAIRS = {
    "checkpoint": ("checkpoint_path", "checkpoint_sha256"),
    "prediction": ("prediction_path", "prediction_sha256"),
    "history": ("history_path", "history_sha256"),
}


@dataclass(frozen=True)
class RunCacheKey:
    """All inputs that can change a physical fold result."""

    config_sha256: str
    split_sha256: str
    label_map_sha256: str
    implementation_sha256: str
    fold: int | None
    seed: int

    @property
    def digest(self) -> str:
        """Return one compact cache directory or manifest identifier."""
        return canonical_sha256(asdict(self))

    def registry_filters(self) -> dict[str, str | int | None]:
        """Return exact registry fields, intentionally excluding Git commit."""
        return {
            "config_sha256": self.config_sha256,
            "split_sha256": self.split_sha256,
            "label_map_sha256": self.label_map_sha256,
            "implementation_sha256": self.implementation_sha256,
            "fold": self.fold,
            "seed": self.seed,
            "status": "completed",
        }


@dataclass(frozen=True)
class CachedRun:
    """A completed registry row whose declared files still match their hashes."""

    row: dict[str, str]
    verified_artifacts: dict[str, str]

    @property
    def run_id(self) -> str:
        return self.row["run_id"]


def _implementation_files(
    paths: tuple[str | Path, ...],
    *,
    root: Path,
    directory_suffixes: tuple[str, ...],
) -> list[Path]:
    files: set[Path] = set()
    for raw_path in paths:
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = root / candidate
        candidate = candidate.resolve()
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise ValueError(f"implementation path is outside project root: {candidate}") from error
        if candidate.is_file():
            files.add(candidate)
        elif candidate.is_dir():
            files.update(
                path.resolve()
                for path in candidate.rglob("*")
                if path.is_file()
                and path.suffix in directory_suffixes
                and "__pycache__" not in path.parts
            )
        else:
            raise FileNotFoundError(f"implementation path does not exist: {candidate}")
    if not files:
        raise ValueError("implementation paths contain no hashable files")
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def implementation_sha256(
    *paths: str | Path,
    root: str | Path = ROOT,
    directory_suffixes: tuple[str, ...] = (".py",),
) -> str:
    """Hash relevant source content and paths, excluding docs discovered in directories."""
    repository = Path(root).resolve()
    files = _implementation_files(
        tuple(paths),
        root=repository,
        directory_suffixes=directory_suffixes,
    )
    manifest: list[dict[str, str]] = []
    for path in files:
        payload = path.read_bytes().replace(b"\r\n", b"\n")
        manifest.append(
            {
                "path": path.relative_to(repository).as_posix(),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return canonical_sha256(manifest)


def build_run_cache_key(
    config: Any,
    *,
    fold: int | None,
    seed: int,
    implementation_paths: tuple[str | Path, ...],
    split_path: str | Path = SPLITS_CSV,
    label_map_path: str | Path = LABEL_MAPS_JSON,
    root: str | Path = ROOT,
) -> RunCacheKey:
    """Build the exact six-part cache identity required by Task 2."""
    if fold is not None and fold < 0:
        raise ValueError("fold must be non-negative or None")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    return RunCacheKey(
        config_sha256=canonical_sha256(config),
        split_sha256=compute_sha256(split_path),
        label_map_sha256=compute_sha256(label_map_path),
        implementation_sha256=implementation_sha256(
            *implementation_paths,
            root=root,
        ),
        fold=fold,
        seed=seed,
    )


def _resolve_artifact(path: str, artifact_root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else artifact_root / candidate


def _verify_row_artifacts(
    row: dict[str, str],
    *,
    required_artifacts: frozenset[str],
    artifact_root: Path,
) -> dict[str, str] | None:
    verified: dict[str, str] = {}
    for artifact, (path_field, hash_field) in ARTIFACT_FIELD_PAIRS.items():
        raw_path = row[path_field]
        expected_hash = row[hash_field]
        if artifact in required_artifacts and (not raw_path or not expected_hash):
            return None
        if bool(raw_path) != bool(expected_hash):
            return None
        if not raw_path:
            continue
        try:
            verified[artifact] = verify_artifact(
                _resolve_artifact(raw_path, artifact_root),
                expected_hash,
            )
        except ArtifactVerificationError:
            return None
    return verified


def find_cached_run(
    registry: RunRegistry,
    key: RunCacheKey,
    *,
    required_artifacts: tuple[str, ...] = ("checkpoint",),
    artifact_root: str | Path = ROOT,
) -> CachedRun | None:
    """Return the newest valid completed run, or ``None`` so the caller reruns it."""
    unknown = set(required_artifacts) - set(ARTIFACT_FIELD_PAIRS)
    if unknown:
        raise KeyError(f"unknown artifact kinds: {sorted(unknown)}")
    candidates = registry.find(**key.registry_filters())
    for index in reversed(candidates.index):
        row = {name: str(value) for name, value in candidates.loc[index].to_dict().items()}
        verified = _verify_row_artifacts(
            row,
            required_artifacts=frozenset(required_artifacts),
            artifact_root=Path(artifact_root),
        )
        if verified is not None:
            return CachedRun(row=row, verified_artifacts=verified)
    return None
