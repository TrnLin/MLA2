"""Deterministic execution and machine provenance for model runs."""

from __future__ import annotations

import os
import platform
import random
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch

from fashion.config import ROOT


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch and select deterministic backend settings."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=False)


def seed_worker(worker_id: int) -> None:
    """Seed one DataLoader worker from PyTorch's worker-specific initial seed."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_torch_generator(seed: int) -> torch.Generator:
    """Create a CPU generator for reproducible sampling and worker seeds."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def capture_git_state(root: str | Path = ROOT) -> dict[str, Any]:
    """Return the current Git commit and whether tracked files differ from it."""
    repository = Path(root)

    def run(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *arguments],
            cwd=repository,
            check=False,
            capture_output=True,
            text=True,
        )

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain", "--untracked-files=no")
    return {
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


def capture_runtime() -> dict[str, Any]:
    """Capture JSON-serializable software and hardware provenance."""
    cuda_available = torch.cuda.is_available()
    gpu: dict[str, Any] | None = None
    if cuda_available:
        properties = torch.cuda.get_device_properties(0)
        gpu = {
            "name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "capability": list(torch.cuda.get_device_capability(0)),
        }
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "processor": platform.processor() or None,
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "total_ram_bytes": int(psutil.virtual_memory().total),
        "packages": {
            package: _package_version(package)
            for package in (
                "numpy",
                "pandas",
                "pillow",
                "scikit-image",
                "scikit-learn",
                "torch",
                "torchvision",
            )
        },
        "cuda_available": cuda_available,
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu": gpu,
    }
