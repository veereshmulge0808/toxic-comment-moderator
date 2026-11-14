import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from src.config import (
    DATA_PROCESSED_DIR,
    LABEL_COLS,
    N_SPLITS,
    RANDOM_SEED,
    TRAIN_FOLDS_PATH,
    TRAIN_RAW_PATH,
)


def create_stratified_folds() -> None:
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading raw training data from {TRAIN_RAW_PATH}...")
    df = pd.read_csv(TRAIN_RAW_PATH)

    # Guard against any NaN text entries in raw CSV
    df["comment_text"] = df["comment_text"].fillna("")
    df["fold"] = -1

    mskf = MultilabelStratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    y = df[LABEL_COLS].values
    for fold, (_, val_idx) in enumerate(mskf.split(df, y)):
        df.loc[val_idx, "fold"] = fold

    assert (df["fold"] != -1).all(), "Error: Some rows were not assigned a fold!"

    # Audit fold balance across all 6 labels
    print("\n=== 5-Fold Multi-Label Stratification Audit (Positive Counts) ===")
    audit_df = df.groupby("fold")[LABEL_COLS].sum()
    audit_df["total_rows"] = df.groupby("fold").size()
    print(audit_df.to_string())

    df.to_csv(TRAIN_FOLDS_PATH, index=False)
    print(f"\n[OK] Saved stratified folds to {TRAIN_FOLDS_PATH} ({len(df):,} rows).")


if __name__ == "__main__":
    create_stratified_folds()