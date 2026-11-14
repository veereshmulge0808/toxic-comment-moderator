from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

TRAIN_RAW_PATH = DATA_RAW_DIR / "train.csv"
TRAIN_FOLDS_PATH = DATA_PROCESSED_DIR / "train_folds.csv"
TEST_CLEAN_PATH = DATA_PROCESSED_DIR / "test_clean.csv"

LABEL_COLS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

N_SPLITS = 5
RANDOM_SEED = 42