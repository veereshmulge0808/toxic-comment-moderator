import json
import re
import time
import zipfile
from collections import Counter

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.config import (
    ARTIFACTS_DIR,
    DATA_DIR,
    LABEL_COLS,
    N_SPLITS,
    RANDOM_SEED,
    TEST_CLEAN_PATH,
    TRAIN_FOLDS_PATH,
)
from src.preprocessing import clean_text

RNN_DIR = ARTIFACTS_DIR / "rnn"
EMBED_DIR = DATA_DIR / "embeddings"
FASTTEXT_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip"

# Hyperparameters grounded in Phase 2 EDA (p95 = 236 words)
MAX_VOCAB_SIZE = 60_000
MAX_LEN = 256
EMBED_DIM = 300
HIDDEN_SIZE = 128
BATCH_SIZE = 128
EPOCHS = 4
LEARNING_RATE = 1e-3

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def seed_everything(seed: int = RANDOM_SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(clean_text(text).lower())


def build_vocab(texts: list[str], max_vocab: int = MAX_VOCAB_SIZE) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(tokenize(text))

    # 0 = <PAD>, 1 = <UNK>
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in counter.most_common(max_vocab - 2):
        vocab[word] = len(vocab)
    return vocab


def encode_texts(texts: list[str], vocab: dict[str, int], max_len: int = MAX_LEN) -> np.ndarray:
    encoded = np.zeros((len(texts), max_len), dtype=np.int64)
    unk_idx = vocab["<UNK>"]
    for i, text in enumerate(texts):
        tokens = tokenize(text)[:max_len]
        for j, tok in enumerate(tokens):
            encoded[i, j] = vocab.get(tok, unk_idx)
    return encoded


def load_or_build_embedding_matrix(vocab: dict[str, int]) -> np.ndarray:
    """Downloads FastText 300d vectors once and extracts rows matching our vocabulary."""
    EMBED_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = EMBED_DIR / f"fasttext_matrix_{len(vocab)}.npy"
    if cache_path.exists():
        print(f"[CACHE] Loading cached FastText embedding matrix from {cache_path}...")
        return np.load(cache_path)

    zip_path = EMBED_DIR / "wiki-news-300d-1M.vec.zip"
    vec_path = EMBED_DIR / "wiki-news-300d-1M.vec"

    if not vec_path.exists():
        if not zip_path.exists():
            print("Downloading FastText wiki-news-300d-1M vectors (~650 MB)...")
            response = requests.get(FASTTEXT_URL, stream=True, timeout=120)
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))
            with open(zip_path, "wb") as f, tqdm(
                desc=zip_path.name,
                total=total_size,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in response.iter_content(chunk_size=8192):
                    bar.update(f.write(chunk))

        print("Extracting FastText .vec file...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(EMBED_DIR)

    rng = np.random.default_rng(RANDOM_SEED)
    matrix = rng.normal(0.0, 0.05, size=(len(vocab), EMBED_DIM)).astype(np.float32)
    matrix[0] = np.zeros(EMBED_DIM, dtype=np.float32)  # <PAD> stays zero

    hits = 0
    print("Matching vocabulary against FastText vectors...")
    with open(vec_path, "r", encoding="utf-8", errors="ignore") as f:
        _ = f.readline()  # Skip header line
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            if word in vocab:
                matrix[vocab[word]] = np.asarray(parts[1:], dtype=np.float32)
                hits += 1

    print(f"Matched {hits:,}/{len(vocab):,} vocab tokens ({hits / len(vocab) * 100:.1f}%) in FastText.")
    np.save(cache_path, matrix)
    return matrix


class ToxicSequenceDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray | None = None):
        self.x = torch.tensor(x, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        if self.y is not None:
            return self.x[idx], self.y[idx]
        return self.x[idx]


class SpatialDropout1D(nn.Module):
    """Drops entire embedding channels across the sequence length dimension."""

    def __init__(self, p: float = 0.25):
        super().__init__()
        self.dropout = nn.Dropout2d(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1).unsqueeze(3)
        x = self.dropout(x)
        return x.squeeze(3).permute(0, 2, 1)


class ToxicBiGRU(nn.Module):
    def __init__(self, embedding_matrix: np.ndarray, hidden_size: int = HIDDEN_SIZE, num_classes: int = 6):
        super().__init__()
        vocab_size, embed_dim = embedding_matrix.shape
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.embedding.weight.requires_grad = True

        self.spatial_dropout = SpatialDropout1D(0.25)
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

        # Dual pooling concatenates BiGRU Max + BiGRU Avg -> 4 * hidden_size
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = x != 0
        empty_rows = mask.sum(dim=1) == 0
        if empty_rows.any():
            mask = mask.clone()
            mask[empty_rows, 0] = True

        emb = self.spatial_dropout(self.embedding(x))
        gru_out, _ = self.gru(emb)

        # 1. Masked Global Max Pooling
        masked_gru = gru_out.masked_fill(~mask.unsqueeze(-1), -1e9)
        max_pool = masked_gru.max(dim=1).values

        # 2. Masked Global Average Pooling
        lengths = mask.sum(dim=1).clamp(min=1).unsqueeze(-1).float()
        avg_pool = (gru_out * mask.unsqueeze(-1).float()).sum(dim=1) / lengths

        features = torch.cat([max_pool, avg_pool], dim=1)
        return self.head(features)


def compute_mean_roc_auc(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, dict[str, float]]:
    per_label = {
        label: float(roc_auc_score(y_true[:, idx], y_pred[:, idx]))
        for idx, label in enumerate(LABEL_COLS)
    }
    return float(np.mean(list(per_label.values()))), per_label


@torch.no_grad()
def predict_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    for batch in loader:
        x_batch = batch[0] if isinstance(batch, (list, tuple)) else batch
        x_batch = x_batch.to(device, non_blocking=True)
        logits = model(x_batch)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds.append(probs)
    return np.vstack(preds)


def main() -> None:
    seed_everything(RANDOM_SEED)
    RNN_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Phase 4 BiGRU Training on device: {device}")
    if device.type == "cpu":
        print("[WARN] CPU detected. Switch to Google Colab T4 GPU for ~20x faster training!")

    train_df = pd.read_csv(TRAIN_FOLDS_PATH)
    test_df = pd.read_csv(TEST_CLEAN_PATH)

    train_texts = train_df["comment_text"].fillna("").tolist()
    test_texts = test_df["comment_text"].fillna("").tolist()
    y_train = train_df[LABEL_COLS].values.astype(np.float32)
    y_test = test_df[LABEL_COLS].values.astype(np.float32)
    folds = train_df["fold"].values

    print(f"Building vocabulary (max_vocab={MAX_VOCAB_SIZE:,}, max_len={MAX_LEN})...")
    vocab = build_vocab(train_texts, MAX_VOCAB_SIZE)
    embedding_matrix = load_or_build_embedding_matrix(vocab)

    x_train = encode_texts(train_texts, vocab, MAX_LEN)
    x_test = encode_texts(test_texts, vocab, MAX_LEN)
    test_loader = DataLoader(ToxicSequenceDataset(x_test), batch_size=BATCH_SIZE * 2, shuffle=False)

    oof_preds = np.zeros((len(train_df), len(LABEL_COLS)), dtype=np.float32)
    test_preds = np.zeros((len(test_df), len(LABEL_COLS)), dtype=np.float32)

    criterion = nn.BCEWithLogitsLoss()
    use_amp = device.type == "cuda"

    print(f"\n=== Training Phase 4 FastText + BiGRU ({N_SPLITS}-Fold CV) ===")
    for fold in range(N_SPLITS):
        t0 = time.perf_counter()
        trn_idx = folds != fold
        val_idx = folds == fold

        trn_loader = DataLoader(
            ToxicSequenceDataset(x_train[trn_idx], y_train[trn_idx]),
            batch_size=BATCH_SIZE,
            shuffle=True,
            pin_memory=use_amp,
        )
        val_loader = DataLoader(
            ToxicSequenceDataset(x_train[val_idx], y_train[val_idx]),
            batch_size=BATCH_SIZE * 2,
            shuffle=False,
            pin_memory=use_amp,
        )

        model = ToxicBiGRU(embedding_matrix, hidden_size=HIDDEN_SIZE).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        best_val_auc = 0.0
        best_val_probs = None
        ckpt_path = RNN_DIR / f"bigru_fold{fold}.pt"

        for epoch in range(1, EPOCHS + 1):
            model.train()
            running_loss = 0.0
            for x_batch, y_batch in trn_loader:
                x_batch = x_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(x_batch)
                    loss = criterion(logits, y_batch)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item() * len(x_batch)

            scheduler.step()
            val_probs = predict_loader(model, val_loader, device)
            val_auc, _ = compute_mean_roc_auc(y_train[val_idx], val_probs)
            avg_loss = running_loss / len(trn_loader.dataset)
            print(f"Fold {fold} | Epoch {epoch}/{EPOCHS} | Loss: {avg_loss:.4f} | Val ROC AUC: {val_auc:.5f}")

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_val_probs = val_probs
                torch.save(model.state_dict(), ckpt_path)

        oof_preds[val_idx] = best_val_probs
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        test_preds += predict_loader(model, test_loader, device) / N_SPLITS
        print(f"--> Fold {fold} Best Val ROC AUC: {best_val_auc:.5f} ({time.perf_counter() - t0:.1f}s)\n")

    oof_mean_auc, oof_per_label = compute_mean_roc_auc(y_train, oof_preds)
    test_mean_auc, test_per_label = compute_mean_roc_auc(y_test, test_preds)

    print("=== Phase 4 Out-of-Fold (OOF) & Test ROC AUC Summary ===")
    summary_df = pd.DataFrame({
        "OOF_ROC_AUC": pd.Series(oof_per_label).round(5),
        "Test_ROC_AUC": pd.Series(test_per_label).round(5),
    })
    print(summary_df.to_string())
    print(f"\n>>> Phase 4 OOF Mean Column-wise ROC AUC:  {oof_mean_auc:.5f} (vs Phase 3: 0.98631)")
    print(f">>> Phase 4 Test Mean Column-wise ROC AUC: {test_mean_auc:.5f} (vs Phase 3: 0.98063)")

    np.save(RNN_DIR / "oof_preds.npy", oof_preds)
    np.save(RNN_DIR / "test_preds.npy", test_preds)
    with open(RNN_DIR / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f)
    print(f"\n[OK] Saved Phase 4 BiGRU checkpoints & OOF predictions to {RNN_DIR}/")


if __name__ == "__main__":
    main()