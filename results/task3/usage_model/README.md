# Final Usage model

Run from the repository root. Install PyTorch if missing:

```bash
./.venv/bin/python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
```

Replace `path/to/image.jpg` below with your image path, then run:

```bash
PYTHONPATH=src ./.venv/bin/python - <<'PY'
import json
from dataclasses import fields
from pathlib import Path

import torch

from fashion.data.images import load_and_transform_image
from fashion.train.config import Task3BaselineConfig
from fashion.train.model import Task3BaselineCNN
from fashion.train.task3_gender_precision import ieee_precision

folder = Path("results/task3/usage_model")
config = json.loads((folder / "config.json").read_text())
stats = json.loads((folder / "normalization.json").read_text())
classes = json.loads((folder / "class_names.json").read_text())
checkpoint = torch.load(folder / "final_epoch.pt", map_location="cpu", weights_only=True)
assert checkpoint["class_names"] == classes
kwargs = {
    field.name: config["base_config"][field.name]
    for field in fields(Task3BaselineConfig)
}
kwargs["channels"] = tuple(kwargs["channels"])
model = Task3BaselineCNN(Task3BaselineConfig(**kwargs))
model.load_state_dict(checkpoint["model_state_dict"], strict=True)
model.eval()

array = load_and_transform_image(
    "path/to/image.jpg",
    image_size=(kwargs["image_height"], kwargs["image_width"]),
    mean=stats["mean"],
    std=stats["std"],
)
inputs = torch.from_numpy(array.transpose(2, 0, 1).copy()).unsqueeze(0)
torch.set_num_threads(2)
with ieee_precision(torch), torch.inference_mode():
    probabilities = model(inputs).softmax(dim=1)[0]
index = int(probabilities.argmax())
print(classes[index], float(probabilities[index]))
PY
```
