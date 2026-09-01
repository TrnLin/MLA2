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
CV_FOLD_SUMMARY_JSON = PROCESSED_DATA_DIR / "cv_fold_summary.json"
DEVELOPMENT_CLASS_SUMMARY_CSV = PROCESSED_DATA_DIR / "development_class_summary.csv"
DEVELOPMENT_IMAGE_PROFILE_JSON = PROCESSED_DATA_DIR / "development_image_profile.json"
TAXONOMY_JSON = PROCESSED_DATA_DIR / "taxonomy.json"

RESULTS_DIR = ROOT / "results"
FIGURE_DIR = RESULTS_DIR / "figures"
DATA_PREPARATION_FIGURE_DIR = FIGURE_DIR / "data_preparation"
EVIDENCE_DIR = RESULTS_DIR / "evidence"
DATA_PREPARATION_EVIDENCE_DIR = EVIDENCE_DIR / "data_preparation"
RUNS_CSV = RESULTS_DIR / "runs.csv"
TASK1_RESULT_DIR = RESULTS_DIR / "task1"
TASK1_FIGURE_DIR = FIGURE_DIR / "task1"
TASK1_EVIDENCE_DIR = EVIDENCE_DIR / "task1"
TASK1_HOG_CACHE_DIR = PROCESSED_DATA_DIR / "task1_hog_cache"

TARGET_COLUMNS = ("articleType", "season", "gender", "usage")
RANDOM_SEED = 2753
DEVELOPMENT_RATIO = 0.85
HOLDOUT_RATIO = 0.15
CV_FOLD_COUNT = 5
