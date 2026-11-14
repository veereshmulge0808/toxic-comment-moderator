# Real-Time Multi-Label Toxic Comment Moderation System

An end-to-end Natural Language Processing and Deep Learning system trained on the Wikipedia Jigsaw Toxic Comment dataset to detect six overlapping categories of online abuse (`toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`) and moderate live messages via a quantized FastAPI microservice.

## System Architecture

1. **Validation & Data Pipeline:** 5-fold `MultilabelStratifiedKFold` cross-validation preserving exact class proportions for ultra-rare categories (`threat` at ~0.30%).
2. **Progressive Modeling Ladder:**
   - **Baseline:** Word `(1,2)` + Character `(2,5)` TF-IDF with One-vs-Rest Logistic Regression.
   - **Sequence Deep Learning:** Pretrained FastText embeddings + Bidirectional GRU with Global Max/Average Dual Pooling.
   - **Transformer Fine-Tuning:** Fine-tuned encoder Transformer (`DeBERTa-v3` / `ModernBERT`) optimized via `BCEWithLogitsLoss`.
3. **Production Optimization:** Post-training dynamic INT8 quantization via ONNX Runtime (`<30ms` CPU inference) paired with Precision-Recall calibrated per-class thresholds.
4. **Deployment:** Containerized FastAPI REST service integrated with a real-time Discord moderation bot / interactive chat UI.

## Setup & Phase 1 Execution

```bash
# 1. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset & filter unscored test labels
python scripts/download_data.py

# 4. Generate deterministic 5-fold stratified split
python -m src.create_folds

## Model Progression Benchmarks

| Phase | Model Architecture | 5-Fold OOF ROC AUC | CPU Latency (ms/msg) | Status |
| :--- | :--- | :---: | :---: | :--- |
| **Phase 3** | Word + Char TF-IDF + Logistic Regression | `0.98631` | `1.98 ms/msg` | Completed |
| **Phase 4** | FastText + BiGRU (Dual Pooling) | *Pending* | *Pending* | In Progress |
| **Phase 5** | Fine-Tuned Transformer (`DeBERTa-v3` / `ModernBERT`) | *Pending* | *Pending* | Planned |
| **Phase 6** | ONNX INT8 Quantized Transformer | *Pending* | *Pending* | Planned |
