from pathlib import Path

from packaging.specifiers import SpecifierSet
from packaging.version import Version

from fashion.config import ROOT


def _constraint_pin(package: str, path: Path) -> str:
    prefix = f"{package}=="
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        requirement = raw_line.split(";", maxsplit=1)[0].strip()
        if requirement.startswith(prefix):
            return requirement.removeprefix(prefix)
    raise AssertionError(f"missing {package} constraint")


def test_torch_constraints_allow_runtime_backend_selection() -> None:
    constraints = ROOT / "requirements/constraints-py312.txt"
    torch_pin = _constraint_pin("torch", constraints)
    vision_pin = _constraint_pin("torchvision", constraints)

    assert torch_pin == "2.13.0"
    assert vision_pin == "0.28.0"
    assert SpecifierSet(f"=={torch_pin}").contains(Version("2.13.0+cu126"))
    assert SpecifierSet(f"=={torch_pin}").contains(Version("2.13.0+cpu"))
    assert SpecifierSet(f"=={vision_pin}").contains(Version("0.28.0+cu126"))
    assert SpecifierSet(f"=={vision_pin}").contains(Version("0.28.0+cpu"))
