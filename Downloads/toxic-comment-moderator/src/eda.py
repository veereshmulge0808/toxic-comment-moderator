import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import ARTIFACTS_DIR, LABEL_COLS, TRAIN_FOLDS_PATH

EDA_DIR = ARTIFACTS_DIR / "eda"


def analyze_label_distribution(df: pd.DataFrame) -> None:
    print("\n" + "=" * 65)
    print("1. CLASS DISTRIBUTION & MULTI-LABEL CARDINALITY")
    print("=" * 65)

    total = len(df)
    label_counts = df[LABEL_COLS].sum().sort_values(ascending=False)
    summary = pd.DataFrame({
        "positive_count": label_counts,
        "percentage (%)": (label_counts / total * 100).round(3),
    })
    print(summary.to_string())

    df["num_labels"] = df[LABEL_COLS].sum(axis=1)
    clean_count = (df["num_labels"] == 0).sum()
    toxic_any_count = (df["num_labels"] > 0).sum()

    print(f"\nClean comments (0 labels):     {clean_count:,} ({clean_count / total * 100:.2f}%)")
    print(f"Toxic comments (>=1 label):    {toxic_any_count:,} ({toxic_any_count / total * 100:.2f}%)")
    print("\nLabels per comment breakdown:")
    print(df["num_labels"].value_counts().sort_index().to_string())

    # Plot class counts
    fig, ax = plt.subplots(figsize=(9, 4.8))
    plot_df = summary.reset_index().rename(columns={"index": "label"})
    sns.barplot(
        data=plot_df,
        x="label",
        y="positive_count",
        hue="label",
        palette="viridis",
        legend=False,
        ax=ax,
    )
    ax.set_title("Jigsaw Training Set: Positive Label Counts (N = 159,571)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Toxicity Category")
    ax.set_ylabel("Number of Comments")
    for container in ax.containers:
        ax.bar_label(container, fmt="%d", padding=3, fontsize=9)
    plt.tight_layout()
    plt.savefig(EDA_DIR / "01_class_distribution.png", dpi=150)
    plt.close()


def analyze_cooccurrence(df: pd.DataFrame) -> None:
    print("\n" + "=" * 65)
    print("2. CONDITIONAL CO-OCCURRENCE MATRIX P(Column | Row)")
    print("=" * 65)

    # Entry (i, j) = P(label_j == 1 | label_i == 1)
    y = df[LABEL_COLS].values
    co_matrix = np.zeros((len(LABEL_COLS), len(LABEL_COLS)))
    for i in range(len(LABEL_COLS)):
        mask = y[:, i] == 1
        co_matrix[i] = y[mask].mean(axis=0)

    co_df = pd.DataFrame(co_matrix, index=LABEL_COLS, columns=LABEL_COLS).round(3)
    print(co_df.to_string())

    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(
        co_df,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        vmin=0.0,
        vmax=1.0,
        cbar_kws={"label": "P(Column Label | Row Label)"},
        ax=ax,
    )
    ax.set_title("Conditional Label Co-Occurrence P(Column | Row)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "02_label_cooccurrence.png", dpi=150)
    plt.close()


def analyze_text_lengths(df: pd.DataFrame) -> None:
    print("\n" + "=" * 65)
    print("3. SEQUENCE LENGTH DIAGNOSTICS (HYPERPARAMETER SIZING)")
    print("=" * 65)

    texts = df["comment_text"].fillna("")
    df["char_len"] = texts.str.len()
    df["word_len"] = texts.str.split().str.len()
    df["is_toxic"] = (df[LABEL_COLS].sum(axis=1) > 0).map({True: "Toxic (>=1 label)", False: "Clean (0 labels)"})

    percentiles = [0.50, 0.75, 0.90, 0.95, 0.99, 1.00]
    stats = pd.DataFrame({
        "char_len": df["char_len"].quantile(percentiles).astype(int),
        "word_len": df["word_len"].quantile(percentiles).astype(int),
    })
    stats.index = [f"p{int(p * 100)}" for p in percentiles]
    print(stats.to_string())

    # Plot truncated word length distribution (up to 300 words for readability)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    clipped = df[df["word_len"] <= 300]
    sns.histplot(
        data=clipped,
        x="word_len",
        hue="is_toxic",
        bins=60,
        stat="density",
        common_norm=False,
        alpha=0.5,
        ax=ax,
    )
    ax.axvline(128, color="red", linestyle="--", linewidth=1.3, label="128 words")
    ax.axvline(256, color="black", linestyle=":", linewidth=1.3, label="256 words")
    ax.set_title("Word Count Distribution: Clean vs. Toxic Comments (<= 300 words)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Word Count")
    ax.legend()
    plt.tight_layout()
    plt.savefig(EDA_DIR / "03_word_length_distribution.png", dpi=150)
    plt.close()


def analyze_noise_and_bias(df: pd.DataFrame) -> None:
    print("\n" + "=" * 65)
    print("4. WIKIPEDIA NOISE ARTIFACTS & IDENTITY BIAS AUDIT")
    print("=" * 65)

    texts = df["comment_text"].fillna("")
    is_toxic = df[LABEL_COLS].sum(axis=1) > 0

    patterns = {
        "IPv4 Addresses (e.g. 192.168.1.1)": r"\b\d{1,3}(?:\.\d{1,3}){3}\b",
        "HTTP/HTTPS URLs": r"https?://\S+|www\.\S+",
        "Repeated Char Spam (4+ same chars)": r"(.)\1{3,}",
        "Obfuscated Profanity (f*ck, sh!t, etc.)": r"\b[a-zA-Z]*[*@$!#0-9]+[a-zA-Z]+\b",
        "Newline / Formatting Noise (\\n)": r"\n",
    }

    noise_rows = []
    for name, regex in patterns.items():
        matches = texts.str.contains(regex, regex=True)
        match_count = int(matches.sum())
        tox_rate = float(is_toxic[matches].mean() * 100) if match_count > 0 else 0.0
        noise_rows.append({
            "artifact_pattern": name,
            "matching_comments": match_count,
            "dataset_%": round(match_count / len(df) * 100, 2),
            "toxic_rate_%": round(tox_rate, 2),
        })
    print(pd.DataFrame(noise_rows).to_string(index=False))

    print("\n--- Identity Term Toxicity Skew (Baseline rate = 10.17%) ---")
    identity_terms = ["gay", "muslim", "jew", "black", "white", "women", "christian", "trans"]
    bias_rows = []
    for term in identity_terms:
        matches = texts.str.contains(rf"\b{term}s?\b", case=False, regex=True)
        count = int(matches.sum())
        tox_rate = float(is_toxic[matches].mean() * 100) if count > 0 else 0.0
        bias_rows.append({
            "identity_term": term,
            "mentions": count,
            "toxic_rate_%": round(tox_rate, 2),
        })
    print(pd.DataFrame(bias_rows).to_string(index=False))


def main() -> None:
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading stratified dataset from {TRAIN_FOLDS_PATH}...")
    df = pd.read_csv(TRAIN_FOLDS_PATH)

    analyze_label_distribution(df)
    analyze_cooccurrence(df)
    analyze_text_lengths(df)
    analyze_noise_and_bias(df)

    print(f"\n[OK] All Phase 2 diagnostic charts saved to {EDA_DIR}/")


if __name__ == "__main__":
    main()