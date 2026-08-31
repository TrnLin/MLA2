from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from fashion.task2.evidence import (
    DEPLOYMENT_CONSUMERS,
    FROZEN_BUNDLE,
    build_file_impact_edges,
    build_task2_evidence,
    plot_file_impact_flow,
    validate_file_impact_edges,
)


def test_file_impact_edges_contain_required_producers_and_consumers() -> None:
    edges = build_file_impact_edges()
    nodes = set(edges["producer"]) | set(edges["consumer"])

    assert {
        "data/processed/splits.csv",
        "data/processed/label_maps.json",
        "fashion.data.torch",
        "fashion.models.season",
        "fashion.train.engine",
        "configs/task2/*.json",
        "results/runs.csv",
        "results/evidence/task2",
        "selection_freeze.json",
        FROZEN_BUNDLE,
        *DEPLOYMENT_CONSUMERS,
    } <= nodes
    assert len(edges) == 18


def test_holdout_cannot_flow_into_training_and_deployment_uses_frozen_bundle() -> None:
    edges = build_file_impact_edges()

    assert not edges.loc[
        edges["producer"].str.contains("holdout", case=False)
        & edges["consumer"].isin(
            {"fashion.data.torch", "fashion.train.engine", "Task 2 runner"}
        )
    ].any(axis=None)
    for consumer in DEPLOYMENT_CONSUMERS:
        assert set(edges.loc[edges["consumer"].eq(consumer), "producer"]) == {
            FROZEN_BUNDLE
        }


def test_validator_rejects_evidence_feedback_or_alternative_app_input() -> None:
    edges = build_file_impact_edges()
    feedback = pd.concat(
        [
            edges,
            pd.DataFrame(
                [
                    {
                        "producer": "results/evidence/task2",
                        "artifact": "preferred result",
                        "consumer": "Task 2 runner",
                        "effect": "unsafe tuning feedback",
                        "phase": "training",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="cannot modify"):
        validate_file_impact_edges(feedback)

    alternative = edges.copy()
    extra = {
        "producer": "Notebook 03",
        "artifact": "mutable choice",
        "consumer": "Streamlit app",
        "effect": "unsafe",
        "phase": "deployment",
    }
    alternative.loc[len(alternative)] = extra
    with pytest.raises(ValueError, match="consume only"):
        validate_file_impact_edges(alternative)


def test_flow_figure_is_nonempty_png_with_all_node_labels(tmp_path: Path) -> None:
    output = tmp_path / "file_impact_flow.png"
    edges = build_file_impact_edges()

    figure, axis = plot_file_impact_flow(edges, output_path=output)
    labels = {text.get_text() for text in axis.texts}

    assert output.is_file()
    assert output.stat().st_size > 50_000
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert set(edges["producer"]) | set(edges["consumer"]) <= labels
    assert "semibold" not in {text.get_fontweight() for text in axis.texts}
    with Image.open(output) as image:
        assert image.width >= 2_500
        assert image.height >= 1_200
    figure.clear()


def test_flow_render_does_not_require_interactive_pyplot(tmp_path: Path, monkeypatch) -> None:
    def reject_interactive_manager(*args, **kwargs):
        raise AssertionError("file-impact render requested an interactive figure manager")

    monkeypatch.setattr("matplotlib.pyplot.subplots", reject_interactive_manager)

    figure, _ = plot_file_impact_flow(
        output_path=tmp_path / "headless-file-impact.png"
    )

    assert (tmp_path / "headless-file-impact.png").is_file()
    figure.clear()


def test_build_task2_evidence_writes_hashed_table_figure_and_manifest(
    tmp_path: Path,
) -> None:
    manifest = build_task2_evidence(
        evidence_directory=tmp_path / "evidence",
        figure_directory=tmp_path / "figures",
    )

    assert manifest["edge_count"] == 18
    assert manifest["holdout_to_training_edges"] == 0
    assert len(manifest["edges_sha256"]) == 64
    assert len(manifest["figure_sha256"]) == 64
    assert not Path(manifest["edges_path"]).is_absolute()
    assert not Path(manifest["figure_path"]).is_absolute()
    assert Path(manifest["manifest_path"]).is_file()
