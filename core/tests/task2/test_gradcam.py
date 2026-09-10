from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from fashion.task2.gradcam import (
    audit_attention_location,
    build_failure_taxonomy,
    compute_gradcam,
    load_gradcam_review_spec,
    plot_gradcam_contact_sheet,
    select_gradcam_examples,
    summarise_failure_taxonomy,
)
from fashion.train.metrics import SEASON_LABELS


def _probabilities(predicted_index: int, confidence: float) -> list[float]:
    remainder = (1.0 - confidence) / (len(SEASON_LABELS) - 1)
    values = [remainder] * len(SEASON_LABELS)
    values[predicted_index] = confidence
    return values


def _selection_inputs() -> tuple[pd.DataFrame, pd.DataFrame, list[int], dict[int, str]]:
    oof_rows: list[dict[str, object]] = []
    context_rows: list[dict[str, object]] = []
    expected_targets: dict[int, str] = {}
    identifiers: list[int] = []
    for label_index, label in enumerate(SEASON_LABELS):
        for offset in range(6):
            identifier = 1000 + label_index * 10 + offset
            identifiers.append(identifier)
            expected_targets[identifier] = label
            correct = offset < 3
            predicted_index = label_index if correct else (label_index + 1) % len(SEASON_LABELS)
            confidence = 0.99 - (offset % 3) * 0.01
            probabilities = _probabilities(predicted_index, confidence)
            for candidate, experiment_id in (
                ("C2", "g3-c2-t0-resnet18"),
                ("I2", "g4-i2-article-type-lambda-0-3-c1"),
            ):
                row: dict[str, object] = {
                    "candidate": candidate,
                    "experiment_id": experiment_id,
                    "seed": 2753,
                    "id": identifier,
                    "fold": identifier % 5,
                    "y_true": label,
                    "y_pred": SEASON_LABELS[predicted_index],
                }
                row.update(
                    {
                        f"prob_{season_label}": probabilities[index]
                        for index, season_label in enumerate(SEASON_LABELS)
                    }
                )
                oof_rows.append(row)
                context_rows.append(
                    {
                        "candidate": candidate,
                        "id": identifier,
                        "run_id": f"{candidate.lower()}-f{identifier % 5}",
                        "path": f"images/{identifier}.jpg",
                        "image_sha256": "a" * 64,
                        "articleType": "Tshirts",
                        "year": 2012,
                        "productDisplayName": f"Product {identifier}",
                        "article_type_shortcut": "conflict" if not correct else "aligned",
                        "acquisition_year": "dominant_2011_2012",
                        "file_size_quartile": "q1_smallest" if offset == 3 else "q3",
                        "product_family_size": "singleton",
                        "image_mode": "rgb",
                    }
                )
    return (
        pd.DataFrame(oof_rows),
        pd.DataFrame(context_rows),
        identifiers,
        expected_targets,
    )


def _small_spec():
    return replace(load_gradcam_review_spec(), expected_row_count=24)


def test_default_gradcam_config_freezes_the_review_boundary() -> None:
    spec = load_gradcam_review_spec()

    assert spec.expected_row_count == 32_753
    assert tuple(candidate.candidate for candidate in spec.candidates) == ("C2", "I2")
    assert spec.examples_per_group == 3
    assert spec.correctness_groups == ("correct", "incorrect")
    assert spec.device == "cpu"
    assert spec.probability_tolerance == 0.0001


def test_gradcam_config_rejects_unknown_fields(tmp_path: Path) -> None:
    payload = json.loads(Path("configs/task2/g6_gradcam_failure_review.json").read_text())
    payload["unexpected"] = True
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="fields changed"):
        load_gradcam_review_spec(path)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("attention_audit", "border_band_fraction", 0.20),
        ("visualisation", "heatmap_colormap", "viridis"),
        (
            "failure_taxonomy",
            "allowed_diagnostic_tags",
            ["model_limitation_or_unmeasured_cause"],
        ),
    ],
)
def test_gradcam_config_rejects_frozen_protocol_drift(
    tmp_path: Path,
    section: str,
    field: str,
    value: object,
) -> None:
    payload = json.loads(Path("configs/task2/g6_gradcam_failure_review.json").read_text())
    payload[section][field] = value
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        load_gradcam_review_spec(path)


def test_gradcam_selection_is_complete_and_deterministic() -> None:
    oof, context, identifiers, targets = _selection_inputs()
    spec = _small_spec()

    selected = select_gradcam_examples(
        oof,
        context,
        spec,
        expected_ids=identifiers,
        expected_targets=targets,
    )

    assert len(selected) == 48
    assert not selected.duplicated(["candidate", "id"]).any()
    assert set(selected.groupby(["candidate", "true_label", "selection_group"]).size()) == {3}
    assert selected.iloc[0]["id"] == 1000
    assert selected.iloc[3]["id"] == 1003
    pd.testing.assert_frame_equal(
        selected,
        select_gradcam_examples(
            oof.sample(frac=1.0, random_state=7),
            context.sample(frac=1.0, random_state=8),
            spec,
            expected_ids=reversed(identifiers),
            expected_targets=targets,
        ),
    )


def test_gradcam_selection_rejects_protected_ids() -> None:
    oof, context, identifiers, targets = _selection_inputs()

    with pytest.raises(ValueError, match="protected IDs"):
        select_gradcam_examples(
            oof,
            context,
            _small_spec(),
            expected_ids=identifiers,
            expected_targets=targets,
            protected_ids={identifiers[0]},
        )


def test_gradcam_selection_rejects_probability_label_drift() -> None:
    oof, context, identifiers, targets = _selection_inputs()
    oof.loc[0, "y_pred"] = "Winter"

    with pytest.raises(ValueError, match="changed predicted labels"):
        select_gradcam_examples(
            oof,
            context,
            _small_spec(),
            expected_ids=identifiers,
            expected_targets=targets,
        )


class _TinyCamModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Conv2d(3, 2, kernel_size=1, bias=False)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(2, 4, bias=False)
        with torch.no_grad():
            self.features.weight.fill_(0.5)
            self.head.weight.zero_()
            self.head.weight[0].fill_(1.0)

    @property
    def gradcam_target_layer(self) -> nn.Module:
        return self.features

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = torch.relu(self.features(images))
        return self.head(torch.flatten(self.pool(features), 1))


class _TinyI2CamModel(_TinyCamModel):
    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = super().forward(images)
        return {"season_logits": logits, "article_type_logits": logits}

    def predict_season_logits(self, images: torch.Tensor) -> torch.Tensor:
        return self.forward(images)["season_logits"]


@pytest.mark.parametrize(
    ("candidate", "model"),
    [("C2", _TinyCamModel()), ("I2", _TinyI2CamModel())],
)
def test_compute_gradcam_returns_bounded_predicted_class_heatmap(
    candidate: str,
    model: nn.Module,
) -> None:
    image = torch.ones((1, 3, 8, 6), dtype=torch.float32)
    model.train()

    result = compute_gradcam(model, image, candidate=candidate, target_index=0)

    assert result.heatmap.shape == (8, 6)
    assert np.isfinite(result.heatmap).all()
    assert 0.0 <= result.heatmap.min() <= result.heatmap.max() <= 1.0
    assert result.probabilities.argmax() == 0
    assert result.activation_shape == (1, 2, 8, 6)
    assert result.gradient_shape == (1, 2, 8, 6)
    assert model.training is True


def test_compute_gradcam_retains_and_flags_zero_heatmap() -> None:
    model = _TinyCamModel()

    result = compute_gradcam(
        model,
        torch.ones((1, 3, 4, 4), dtype=torch.float32),
        candidate="C2",
        target_index=1,
    )

    assert result.zero_heatmap is True
    assert np.count_nonzero(result.heatmap) == 0


def test_attention_audit_flags_border_focus_but_not_foreground_focus() -> None:
    spec = _small_spec().attention
    rgb = np.ones((10, 10, 3), dtype=np.float32)
    rgb[3:7, 3:7] = 0.0
    content = np.ones((10, 10), dtype=bool)
    foreground_heat = np.zeros((10, 10), dtype=float)
    foreground_heat[3:7, 3:7] = 1.0
    border_heat = np.zeros((10, 10), dtype=float)
    border_heat[[0, 1, 8, 9], :] = 1.0
    border_heat[:, [0, 1, 8, 9]] = 1.0

    foreground = audit_attention_location(foreground_heat, rgb, content, spec)
    border = audit_attention_location(border_heat, rgb, content, spec)

    assert foreground["foreground_attention_share"] == pytest.approx(1.0)
    assert foreground["attention_review_flag"] is False
    assert border["border_attention_lift"] > spec.border_attention_lift_review_threshold
    assert border["attention_review_flag"] is True


def test_failure_taxonomy_uses_non_causal_priority_and_real_ids() -> None:
    oof, context, identifiers, targets = _selection_inputs()
    spec = _small_spec()
    selected = select_gradcam_examples(
        oof,
        context,
        spec,
        expected_ids=identifiers,
        expected_targets=targets,
    )
    selected["border_attention_lift"] = 1.5
    selected["foreground_attention_lift"] = 0.6
    selected["attention_review_flag"] = True

    taxonomy = build_failure_taxonomy(selected, spec)
    summary = summarise_failure_taxonomy(taxonomy)

    assert len(taxonomy) == 24
    assert taxonomy["causal_claim_allowed"].eq(False).all()
    assert taxonomy["human_label_ambiguity_review_required"].eq(True).all()
    assert set(taxonomy["id"]) <= set(identifiers)
    assert taxonomy.iloc[0]["primary_failure_hypothesis"] == "weak_data_proxy"
    assert "article_type_shortcut_conflict" in taxonomy.iloc[0]["diagnostic_tags"]
    assert "model_limitation_or_unmeasured_cause" not in taxonomy.iloc[0]["diagnostic_tags"]
    assert summary["selected_error_count"].sum() == 24


def test_gradcam_contact_sheet_writes_all_fixed_examples(tmp_path: Path) -> None:
    oof, context, identifiers, targets = _selection_inputs()
    spec = _small_spec()
    selected = select_gradcam_examples(
        oof,
        context,
        spec,
        expected_ids=identifiers,
        expected_targets=targets,
    )
    c2 = selected.loc[selected["candidate"].eq("C2")]
    overlays = {
        ("C2", int(identifier)): (
            np.ones((8, 6, 3), dtype=np.float32),
            np.linspace(0.0, 1.0, 48, dtype=np.float32).reshape(8, 6),
        )
        for identifier in c2["id"]
    }
    path = tmp_path / "contact.png"

    result = plot_gradcam_contact_sheet(
        selected,
        overlays,
        path,
        candidate="C2",
        spec=spec,
    )

    assert result == path
    assert path.stat().st_size > 20_000
