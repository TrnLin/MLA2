import os
from pathlib import Path

PROJECT_ROOT_ENV = "FASHION_PROJECT_ROOT"


def _resolve_project_root() -> Path:
    """Resolve repository data paths for editable and normal wheel installs."""
    configured = os.environ.get(PROJECT_ROOT_ENV, "").strip()
    if configured:
        return Path(configured).expanduser().resolve()

    package_candidate = Path(__file__).resolve().parents[2]
    if (package_candidate / "pyproject.toml").is_file():
        return package_candidate

    working = Path.cwd().resolve()
    for candidate in (working, *working.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src/fashion").is_dir():
            return candidate
    raise RuntimeError(
        "cannot locate the fashion project root; set FASHION_PROJECT_ROOT to the checkout"
    )


ROOT = _resolve_project_root()

RAW_DATA_DIR = ROOT / "data/raw"
PROCESSED_DATA_DIR = ROOT / "data/processed"

ORIGINAL_DATA_DIR = RAW_DATA_DIR / "original"
ORIGINAL_CSV = ORIGINAL_DATA_DIR / "styles.csv"
ORIGINAL_IMAGE_LINKS_CSV = ORIGINAL_DATA_DIR / "images.csv"
ORIGINAL_IMAGE_DIR = ORIGINAL_DATA_DIR / "images"
ORIGINAL_STYLE_JSON_DIR = ORIGINAL_DATA_DIR / "styles"

TEACHER_DATA_DIR = RAW_DATA_DIR / "teacher"
TEACHER_TRAIN_DIR = TEACHER_DATA_DIR / "train"
TEACHER_TRAIN_CSV = TEACHER_TRAIN_DIR / "styles_train.csv"
TEACHER_TRAIN_IMAGE_DIR = TEACHER_TRAIN_DIR / "images_train"
TEACHER_TEST_DIR = TEACHER_DATA_DIR / "test"
TEST_CSV = TEACHER_TEST_DIR / "styles_prediction.csv"
TEST_IMAGE_DIR = TEACHER_TEST_DIR / "images_test"

AUDIT_DIR = PROCESSED_DATA_DIR / "audit"
PREDICTION_MANIFEST_CSV = PROCESSED_DATA_DIR / "prediction_manifest.csv"
PRODUCT_FAMILIES_CSV = AUDIT_DIR / "product_family_groups.csv.gz"
LABEL_MAPS_JSON = PROCESSED_DATA_DIR / "label_maps.json"
SPLITS_CSV = PROCESSED_DATA_DIR / "splits.csv"
SPLIT_SUMMARY_JSON = PROCESSED_DATA_DIR / "split_summary.json"
DEVELOPMENT_CLASS_SUMMARY_CSV = PROCESSED_DATA_DIR / "development_class_summary.csv"
TAXONOMY_JSON = PROCESSED_DATA_DIR / "taxonomy.json"
ORIGINAL_ONLY_NORMALIZATION_JSON = PROCESSED_DATA_DIR / "normalization_original_only.json"
PAIRED_NORMALIZATION_JSON = PROCESSED_DATA_DIR / "paired_normalization.json"
# The doubled original/high-resolution policy is the main training path.
NORMALIZATION_JSON = PAIRED_NORMALIZATION_JSON

RESULTS_DIR = ROOT / "results"
FIGURE_DIR = RESULTS_DIR / "figures"
EDA_FIGURE_DIR = FIGURE_DIR / "eda"
EVIDENCE_DIR = RESULTS_DIR / "evidence"
EDA_EVIDENCE_DIR = EVIDENCE_DIR / "eda"

TARGET_COLUMNS = ("articleType", "season", "gender", "usage")
RANDOM_SEED = 2753
# Numpy shape order: height, width. The native 60x80 images fill 96x128 exactly.
IMAGE_SIZE = (128, 96)
LEGACY_IMAGE_SIZE = (128, 128)
PAD_COLOR = (255, 255, 255)
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
HOLDOUT_RATIO = 0.15
MIN_SUPPORTED_GROUPS = 3
# Fixed before allocation so the required visual-only season example is never
# selected from validation or protected holdout outcomes.
TRAIN_ONLY_EDA_EXAMPLE_IDS = (41303, 43587)
