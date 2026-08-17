from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

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
FIGURE_DIR = ROOT / "results/figures"
EDA_OUTPUT_DIR = FIGURE_DIR / "eda"
EDA_SAMPLE_SIZE = 2048

# Compatibility aliases used by current callers.
TRAIN_CSV = TEACHER_TRAIN_CSV
TRAIN_IMAGE_DIR = TEACHER_TRAIN_IMAGE_DIR

TARGET_COLUMNS = ("articleType", "season", "gender", "usage")
RANDOM_SEED = 2753
