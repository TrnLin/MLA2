# Train with the combined teacher + rare dataset

Select this dataset version in the same Usage training code. Both sources use `usage` and the
existing nine-class label map. The normal `FashionDataset` returns both teacher and added images.

From the repository root:

```python
from pathlib import Path

from fashion.data import (
    FashionDataset,
    get_cv_split,
    get_samples,
    load_label_maps,
    load_splits,
)
from fashion.data.images import load_and_transform_image

folder = Path("data/processed/teacher_plus_rare_usage_20260906")
splits = load_splits(folder / "splits.csv")
labels = load_label_maps(folder / "label_maps.json")["usage"]["label_to_index"]

train_rows, val_rows = get_cv_split(splits, validation_fold=0)
train_rows = get_samples(train_rows, target="usage")
val_rows = get_samples(val_rows, target="usage")


def transform(path):
    # Height 80, width 60. RGB float32 in [0, 1], channels first.
    return load_and_transform_image(path, image_size=(80, 60)).transpose(2, 0, 1)


training = FashionDataset(train_rows, transform=transform, targets=("usage",))
validation = FashionDataset(val_rows, transform=transform, targets=("usage",))
sample = training[0]
image, target = sample["image"], labels[sample["usage"]]
assert image.shape == (3, 80, 60)
assert len(labels) == 9
print(len(training), len(validation))  # 26315, 6577
```

Keep the existing model's nine outputs and Usage loss. Encode every sample with the same `labels`
mapping. Select folds 0–4 from this full split; do not split again inside a notebook. The exported
`train.csv` and `validation.csv` are the fold-0 rows shown above.

Use a new run/config entry that records this dataset's path and split hash. Old teacher-only run
artifacts and checkpoints do not describe this expanded dataset. If the model fits channel means
and standard deviations, recompute them on its combined training fold only. The loading example
above uses range scaling, which requires no fitted statistics.

Added images supply only Usage labels. Train gender, season and articleType only on rows with
their corresponding valid-label mask. Keep the teacher holdout sealed until final evaluation.
The existing training runtime can wrap this framework-neutral loader in its normal batching code;
PyTorch itself was unavailable in the local environment.
