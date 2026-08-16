from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAIN_CSV = ROOT / "data/train/styles_train.csv"
TRAIN_IMAGE_DIR = ROOT / "data/train/images_train"
TEST_CSV = ROOT / "data/test/styles_prediction.csv"
TEST_IMAGE_DIR = ROOT / "data/test/images_test"
FIGURE_DIR = ROOT / "results/figures"

TARGET_COLUMNS = ("articleType", "season", "gender", "usage")
RANDOM_SEED = 2753
