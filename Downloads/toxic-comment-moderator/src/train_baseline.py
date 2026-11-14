import time
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion, Pipeline

from src.config import (
    ARTIFACTS_DIR,
    LABEL_COLS,
    N_SPLITS,
    RANDOM_SEED,
    TEST_CLEAN_PATH,
    TRAIN_FOLDS_PATH,
)
from src.preprocessing import clean_text

BASELINE_DIR = ARTIFACTS_DIR / "baseline"


def build_pipeline() -> Pipeline:
    """Builds a dual Word + Character TF-IDF + One-vs-Rest Logistic Regression pipeline."""
    features = FeatureUnion(
        transformer_list=[
            (
                "word_tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=100_000,
                    sublinear_tf=True,
                    strip_accents="unicode",
                    token_pattern=r"\w{1,}",
                    dtype=np.float32,
                ),
            ),
            (
                "char_tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(2, 5),
                    max_features=150_000,
                    sublinear_tf=True,
                    dtype=np.float32,
                ),
            ),
        ]
    )

    classifier = OneVsRestClassifier(
        LogisticRegression(
            C=2.0,
            solver="liblinear",
            max_iter=1000,
            random_state=RANDOM_SEED,
        ),
        n_jobs=1,
    )

    return Pipeline([("features", features), ("clf", classifier)])


def compute_mean_roc_auc(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, dict[str, float]]:
    per_label = {}
    for idx, label in enumerate(LABEL_COLS):
        per_label[label] = float(roc_auc_score(y_true[:, idx], y_pred[:, idx]))
    mean_auc = float(np.mean(list(per_label.values())))
    return mean_auc, per_label


def benchmark_latency(pipeline: Pipeline, sample_texts: list[str]) -> float:
    """Measures average single-message CPU inference latency in milliseconds."""
    # Warmup
    for text in sample_texts[:10]:
        pipeline.predict_proba([text])

    start = time.perf_counter()
    for text in sample_texts:
        pipeline.predict_proba([text])
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return elapsed_ms / len(sample_texts)


def main() -> None:
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading 5-fold stratified training set and clean test set...")
    train_df = pd.read_csv(TRAIN_FOLDS_PATH)
    test_df = pd.read_csv(TEST_CLEAN_PATH)

    train_texts = train_df["comment_text"].fillna("").map(clean_text).values
    test_texts = test_df["comment_text"].fillna("").map(clean_text).values

    y_train = train_df[LABEL_COLS].values
    y_test = test_df[LABEL_COLS].values
    folds = train_df["fold"].values

    oof_preds = np.zeros((len(train_df), len(LABEL_COLS)), dtype=np.float32)
    test_preds = np.zeros((len(test_df), len(LABEL_COLS)), dtype=np.float32)

    print(f"\n=== Training Phase 3 Baseline ({N_SPLITS}-Fold CV) ===")
    for fold in range(N_SPLITS):
        t0 = time.perf_counter()
        trn_idx = folds != fold
        val_idx = folds == fold

        pipeline = build_pipeline()
        pipeline.fit(train_texts[trn_idx], y_train[trn_idx])

        val_probs = pipeline.predict_proba(train_texts[val_idx])
        oof_preds[val_idx] = val_probs
        test_preds += pipeline.predict_proba(test_texts) / N_SPLITS

        fold_auc, _ = compute_mean_roc_auc(y_train[val_idx], val_probs)
        dt = time.perf_counter() - t0
        print(f"Fold {fold} | Val Mean ROC AUC: {fold_auc:.5f} | Time: {dt:.1f}s")

    oof_mean_auc, oof_per_label = compute_mean_roc_auc(y_train, oof_preds)
    test_mean_auc, test_per_label = compute_mean_roc_auc(y_test, test_preds)

    print("\n=== Phase 3 Out-of-Fold (OOF) & Test ROC AUC Summary ===")
    summary_df = pd.DataFrame({
        "OOF_ROC_AUC": pd.Series(oof_per_label).round(5),
        "Test_ROC_AUC": pd.Series(test_per_label).round(5),
    })
    print(summary_df.to_string())
    print(f"\n>>> 5-Fold OOF Mean Column-wise ROC AUC:  {oof_mean_auc:.5f}")
    print(f">>> Holdout Test Mean Column-wise ROC AUC: {test_mean_auc:.5f}")

    # Fit final full-data pipeline for standalone baseline artifact & latency check
    print("\nFitting full-dataset baseline pipeline for latency benchmark & export...")
    full_pipeline = build_pipeline()
    full_pipeline.fit(train_texts, y_train)

    latency_ms = benchmark_latency(full_pipeline, test_texts[:200].tolist())
    print(f">>> Single-Message CPU Inference Latency:  {latency_ms:.2f} ms/msg")

    np.save(BASELINE_DIR / "oof_preds.npy", oof_preds)
    np.save(BASELINE_DIR / "test_preds.npy", test_preds)
    joblib.dump(full_pipeline, BASELINE_DIR / "tfidf_logreg_pipeline.joblib")
    print(f"\n[OK] Saved Phase 3 baseline model & OOF predictions to {BASELINE_DIR}/")


if __name__ == "__main__":
    main()