from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

ORIGINAL_CSV = ROOT / "data/train/styles.csv"
ORIGINAL_IMAGE_LINKS_CSV = ROOT / "data/train/images.csv"
ORIGINAL_IMAGE_DIR = ROOT / "data/train/images"
ORIGINAL_STYLE_JSON_DIR = ROOT / "data/train/styles"

TEACHER_TRAIN_CSV = ROOT / "data/train-old/styles_train.csv"
TEACHER_TRAIN_IMAGE_DIR = ROOT / "data/train-old/images_train"
TEST_CSV = ROOT / "data/test/styles_prediction.csv"
TEST_IMAGE_DIR = ROOT / "data/test/images_test"
FIGURE_DIR = ROOT / "results/figures"

# Compatibility aliases used by the existing official-subset EDA.
TRAIN_CSV = TEACHER_TRAIN_CSV
TRAIN_IMAGE_DIR = TEACHER_TRAIN_IMAGE_DIR

TARGET_COLUMNS = ("articleType", "season", "gender", "usage")
RANDOM_SEED = 2753
