import os
from pathlib import Path
import pandas as pd
import requests
from tqdm import tqdm

BASE_URL = (
    "https://huggingface.co/datasets/thesofakillers/"
    "jigsaw-toxic-comment-classification-challenge/resolve/main"
)
FILES = ["train.csv", "test.csv", "test_labels.csv"]

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def download_file(url: str, dest_path: Path) -> None:
    if dest_path.exists():
        print(f"[SKIP] {dest_path} already exists.")
        return

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))

    with open(dest_path, "wb") as f, tqdm(
        desc=dest_path.name,
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            size = f.write(chunk)
            bar.update(size)


def build_clean_test_set() -> None:
    clean_test_path = PROCESSED_DIR / "test_clean.csv"
    if clean_test_path.exists():
        print(f"[SKIP] {clean_test_path} already exists.")
        return

    test_df = pd.read_csv(RAW_DIR / "test.csv")
    labels_df = pd.read_csv(RAW_DIR / "test_labels.csv")

    merged = pd.merge(test_df, labels_df, on="id", how="inner")
    # Jigsaw marked unscored test rows with -1 across all label columns
    clean_test = merged[merged["toxic"] != -1].reset_index(drop=True)
    clean_test.to_csv(clean_test_path, index=False)
    print(
        f"[OK] Saved {clean_test_path} with {len(clean_test):,} scored test comments "
        f"(filtered out {len(merged) - len(clean_test):,} unscored rows)."
    )


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for filename in FILES:
        url = f"{BASE_URL}/{filename}"
        download_file(url, RAW_DIR / filename)

    build_clean_test_set()
    print("\nDataset setup complete!")


if __name__ == "__main__":
    main()