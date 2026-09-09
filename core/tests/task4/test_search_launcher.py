from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from PIL import UnidentifiedImageError

from fashion.config import ROOT
from fashion.task4.search import CropBox

LAUNCHER = ROOT / "scripts/task4/search_one_image.py"


def _load_launcher() -> ModuleType:
    spec = importlib.util.spec_from_file_location("task4_search_one_image_script", LAUNCHER)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Task 4 search launcher")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _response() -> SimpleNamespace:
    rows = [
        {
            "rank": 1,
            "candidate_id": 20,
            "distance": 0.125,
            "articleType": "Tshirts",
            "baseColour": "Blue",
            "productDisplayName": "Blue tee",
            "grade": None,
        },
        {
            "rank": 2,
            "candidate_id": 30,
            "distance": 0.25,
            "articleType": "Jeans",
            "baseColour": "Black",
            "productDisplayName": "Black jeans",
            "grade": None,
        },
    ]
    hits = tuple(SimpleNamespace(to_dict=lambda row=row: row) for row in rows)
    query = SimpleNamespace(warnings=("unusual_aspect_ratio",))
    record = SimpleNamespace(
        query_key="outside-abc123-k5",
        top_k=5,
        query=query,
        results=hits,
    )
    return SimpleNamespace(record=record)


def test_launcher_runs_one_cropped_outside_search_and_prints_ordered_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    launcher = _load_launcher()
    image_path = tmp_path / "outside.png"
    image_path.write_bytes(b"synthetic image")
    fake_bundle = object()
    response = _response()
    png_path = tmp_path / "figures/outside-abc123-k5.png"
    json_path = tmp_path / "evidence/outside-abc123-k5.json"
    calls: dict[str, object] = {}

    def fake_loader(**kwargs: object) -> object:
        calls["load"] = kwargs
        return fake_bundle

    def fake_search(bundle: object, **kwargs: object) -> SimpleNamespace:
        calls["search"] = (bundle, kwargs)
        return response

    def fake_writer(
        received: object,
        **kwargs: object,
    ) -> tuple[Path, Path]:
        calls["write"] = (received, kwargs)
        return png_path, json_path

    monkeypatch.setattr(launcher, "load_search_bundle", fake_loader)
    monkeypatch.setattr(launcher, "run_search", fake_search)
    monkeypatch.setattr(launcher, "write_search_outputs", fake_writer)

    exit_code = launcher.main(
        [
            "--image",
            str(image_path),
            "--crop",
            "10",
            "5",
            "70",
            "75",
            "--top-k",
            "5",
            "--rating",
            "good",
            "--note",
            "all five matches are useful",
        ]
    )

    assert exit_code == 0
    assert calls["load"] == {
        "model_package": launcher.MODEL_PACKAGE,
        "gallery_directory": launcher.GALLERY_DIRECTORY,
        "splits_path": launcher.SPLITS_PATH,
        "device": "cpu",
    }
    assert calls["search"] == (
        fake_bundle,
        {
            "image_path": image_path,
            "query_id": None,
            "crop": CropBox(10, 5, 70, 75),
            "top_k": 5,
            "rating": "good",
            "note": "all five matches are useful",
        },
    )
    assert calls["write"] == (
        response,
        {
            "figure_directory": launcher.FIGURE_DIRECTORY,
            "evidence_directory": launcher.EVIDENCE_DIRECTORY,
        },
    )
    assert json.loads(capsys.readouterr().out) == {
        "query_key": "outside-abc123-k5",
        "top_k": 5,
        "png_path": str(png_path),
        "json_path": str(json_path),
        "warnings": ["unusual_aspect_ratio"],
        "results": [
            {
                "rank": 1,
                "candidate_id": 20,
                "distance": 0.125,
                "articleType": "Tshirts",
                "baseColour": "Blue",
                "productDisplayName": "Blue tee",
                "grade": None,
            },
            {
                "rank": 2,
                "candidate_id": 30,
                "distance": 0.25,
                "articleType": "Jeans",
                "baseColour": "Black",
                "productDisplayName": "Black jeans",
                "grade": None,
            },
        ],
    }


def test_launcher_passes_known_query_device_and_output_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    launcher = _load_launcher()
    response = _response()
    figure_directory = tmp_path / "my-figures"
    evidence_directory = tmp_path / "my-evidence"
    calls: dict[str, object] = {}

    def fake_loader(**kwargs: object) -> object:
        calls["load"] = kwargs
        return "bundle"

    def fake_search(bundle: object, **kwargs: object) -> SimpleNamespace:
        calls["search"] = (bundle, kwargs)
        return response

    def fake_writer(received: object, **kwargs: object) -> tuple[Path, Path]:
        calls["write"] = (received, kwargs)
        return figure_directory / "result.png", evidence_directory / "result.json"

    monkeypatch.setattr(launcher, "load_search_bundle", fake_loader)
    monkeypatch.setattr(launcher, "run_search", fake_search)
    monkeypatch.setattr(launcher, "write_search_outputs", fake_writer)

    assert (
        launcher.main(
            [
                "--query-id",
                "10",
                "--device",
                "cuda",
                "--figure-directory",
                str(figure_directory),
                "--evidence-directory",
                str(evidence_directory),
            ]
        )
        == 0
    )

    assert calls["load"]["device"] == "cuda"
    assert calls["search"] == (
        "bundle",
        {
            "image_path": None,
            "query_id": 10,
            "crop": None,
            "top_k": 5,
            "rating": None,
            "note": None,
        },
    )
    assert calls["write"] == (
        response,
        {
            "figure_directory": figure_directory,
            "evidence_directory": evidence_directory,
        },
    )
    capsys.readouterr()


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["--query-id", "10", "--image", "outside.png"],
        ["--image", "outside.png", "--top-k", "0"],
        ["--image", "outside.png", "--top-k", "21"],
        ["--image", "outside.png", "--crop", "0", "0", "20"],
    ],
    ids=["no-source", "both-sources", "k-zero", "k-twenty-one", "partial-crop"],
)
def test_launcher_parser_rejects_invalid_query_shapes_with_code_two(
    argv: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    launcher = _load_launcher()

    with pytest.raises(SystemExit) as caught:
        launcher._parser().parse_args(argv)

    assert caught.value.code == 2
    error = json.loads(capsys.readouterr().err)
    assert set(error) == {"error"}
    assert error["error"]


@pytest.mark.parametrize(
    ("rating", "note"),
    [("good", None), (None, "useful")],
    ids=["rating-without-note", "note-without-rating"],
)
def test_launcher_returns_json_error_for_incomplete_outside_feedback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    rating: str | None,
    note: str | None,
) -> None:
    launcher = _load_launcher()
    image_path = tmp_path / "outside.png"
    argv = ["--image", str(image_path)]
    if rating is not None:
        argv.extend(["--rating", rating])
    if note is not None:
        argv.extend(["--note", note])

    monkeypatch.setattr(launcher, "load_search_bundle", lambda **kwargs: object())
    monkeypatch.setattr(
        launcher,
        "run_search",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("outside rating and note must both be provided")
        ),
    )

    assert launcher.main(argv) == 2
    assert json.loads(capsys.readouterr().err) == {
        "error": "outside rating and note must both be provided"
    }


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError("missing image"),
        UnidentifiedImageError("bad image"),
        FloatingPointError("bad embedding"),
        RuntimeError("render failed"),
        ValueError("invalid crop"),
    ],
    ids=["file", "image", "numeric", "runtime", "value"],
)
def test_launcher_converts_expected_failures_to_json_stderr(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error: Exception,
) -> None:
    launcher = _load_launcher()
    monkeypatch.setattr(
        launcher,
        "load_search_bundle",
        lambda **kwargs: (_ for _ in ()).throw(error),
    )

    assert launcher.main(["--query-id", "10"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err) == {"error": str(error)}


def test_launcher_help_states_that_protected_data_stays_closed() -> None:
    completed = subprocess.run(
        [sys.executable, str(LAUNCHER), "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "development fold" in completed.stdout
    assert "never opens holdout, quarantine, or official teacher-test images" in completed.stdout


def test_launcher_has_one_main_guard_that_exits_with_main_result() -> None:
    tree = ast.parse(LAUNCHER.read_text(encoding="utf-8"))
    guards = [
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
        and any(
            isinstance(comparator, ast.Constant) and comparator.value == "__main__"
            for comparator in node.test.comparators
        )
    ]

    assert len(guards) == 1
    assert ast.dump(guards[0].body[0]) == ast.dump(
        ast.Raise(
            exc=ast.Call(
                func=ast.Name(id="SystemExit", ctx=ast.Load()),
                args=[
                    ast.Call(
                        func=ast.Name(id="main", ctx=ast.Load()),
                        args=[],
                        keywords=[],
                    )
                ],
                keywords=[],
            )
        )
    )
