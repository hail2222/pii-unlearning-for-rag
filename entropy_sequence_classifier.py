"""
Entropy Sequence Classification for PII Detection.
Increasingly complex sequence classifiers using ONLY token-level entropy
(no hidden states required — works in API-only settings).

Methods (in order of complexity):
  1. Logistic Regression     — hand-crafted features
  2. Random Forest           — hand-crafted features, non-linear
  3. XGBoost                 — hand-crafted features, gradient boosting
  4. LSTM                    — raw entropy sequence
  5. GRU                     — raw entropy sequence
  6. 1D CNN                  — raw entropy sequence (baseline)
  7. Multi-scale CNN         — parallel kernels (3, 5, 7)
  8. 2-channel CNN           — entropy + delta entropy
  9. CNN + LSTM              — local patterns → sequential memory
  10. Transformer encoder    — self-attention over entropy sequence

Ground truth: A_located = Positive, A_not_located + B + C = Negative
Train/test: 80/20 sample-level split (seed=42, no data leakage).

Usage:
    python entropy_sequence_classifier.py --results-dir results_llama8b_20260401_1429
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import pickle
import math
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from run_experiment import SampleResult
from config import RESULTS_DIR as _DEFAULT_RESULTS_DIR, MAX_NEW_TOKENS

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

MAX_LEN = MAX_NEW_TOKENS


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=str, default=_DEFAULT_RESULTS_DIR)
    p.add_argument("--test-ratio",  type=float, default=0.2)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--epochs",      type=int,   default=30)
    p.add_argument("--batch-size",  type=int,   default=32)
    p.add_argument("--device",      type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ── Data utils ─────────────────────────────────────────────────────────────────

def load(results_dir, condition):
    with open(os.path.join(results_dir, f"{condition}.pkl"), "rb") as f:
        return pickle.load(f)


def split_samples(samples, test_ratio, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(samples))
    n_test = max(1, int(len(samples) * test_ratio))
    test_set = set(idx[:n_test])
    return ([s for i, s in enumerate(samples) if i not in test_set],
            [s for i, s in enumerate(samples) if i in test_set])


def pad_seq(seq, max_len):
    seq = np.array(seq, dtype=np.float32)
    if len(seq) >= max_len:
        return seq[:max_len]
    return np.pad(seq, (0, max_len - len(seq)), constant_values=0.0)


def delta_seq(seq, max_len):
    s = np.array(seq, dtype=np.float32)
    d = np.abs(np.diff(s, prepend=s[0] if len(s) > 0 else 0.0))
    return pad_seq(d, max_len)


# ── Feature extraction ─────────────────────────────────────────────────────────

def extract_features(r: SampleResult) -> np.ndarray:
    seq = np.array(r.entropy_seq, dtype=np.float32)
    if len(seq) == 0:
        return np.zeros(12, dtype=np.float32)
    delta = np.abs(np.diff(seq, prepend=seq[0]))
    below = (seq < 0.5).astype(float)
    max_run = cur = 0
    for v in below:
        cur = cur + 1 if v else 0
        max_run = max(max_run, cur)
    return np.array([
        seq.mean(), seq.std(), seq.min(), seq.max(),
        np.percentile(seq, 25), np.median(seq),
        below.mean(), float(max_run),
        delta.mean(), delta.max(),
        seq[:10].mean() if len(seq) >= 10 else seq.mean(),
        seq[-10:].mean() if len(seq) >= 10 else seq.mean(),
    ], dtype=np.float32)


# ── Output helpers ─────────────────────────────────────────────────────────────

def print_config(config: dict):
    items = ", ".join(f"{k}={v}" for k, v in config.items())
    print(f"  Config     : {items}")


def compute_and_print(y_true, y_pred):
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    n_pos, n_neg = int(y_true.sum()), int((y_true == 0).sum())
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    print(f"  Positive ✓ (TP): {tp}/{n_pos} ({tp/n_pos*100:.1f}%)")
    print(f"  Positive ✗ (FN): {fn}/{n_pos} ({fn/n_pos*100:.1f}%)")
    print(f"  Negative ✓ (TN): {tn}/{n_neg} ({tn/n_neg*100:.1f}%)")
    print(f"  Negative ✗ (FP): {fp}/{n_neg} ({fp/n_neg*100:.1f}%)")
    print(f"  Accuracy : {acc:.4f}  Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}")
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1,
            "TP": tp, "FN": fn, "FP": fp, "TN": tn}


def section(title):
    print(f"\n{'='*62}")
    print(f"  {title}")
    print(f"{'='*62}")


def sklearn_cv(pipe, X, y, n_splits=5):
    k = min(n_splits, int(y.sum()), int((y == 0).sum()))
    if k < 2:
        return
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    s = cross_validate(pipe, X, y, cv=cv, scoring=["accuracy", "f1", "roc_auc"])
    print(f"  Train CV   : Accuracy={s['test_accuracy'].mean():.3f}±{s['test_accuracy'].std():.3f}"
          f"  F1={s['test_f1'].mean():.3f}±{s['test_f1'].std():.3f}"
          f"  AUC={s['test_roc_auc'].mean():.3f}±{s['test_roc_auc'].std():.3f}")


# ── Dataset ────────────────────────────────────────────────────────────────────

class SeqDataset(Dataset):
    """Single-channel entropy sequence dataset."""
    def __init__(self, pos, neg, max_len):
        seqs, labels = [], []
        for r in pos:
            seqs.append(pad_seq(r.entropy_seq, max_len)); labels.append(1)
        for r in neg:
            seqs.append(pad_seq(r.entropy_seq, max_len)); labels.append(0)
        self.X = torch.tensor(np.stack(seqs), dtype=torch.float32).unsqueeze(1)  # (N,1,L)
        self.y = torch.tensor(labels, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


class TwoChannelDataset(Dataset):
    """Two-channel dataset: entropy + delta entropy."""
    def __init__(self, pos, neg, max_len):
        seqs, labels = [], []
        for r in pos:
            e = pad_seq(r.entropy_seq, max_len)
            d = delta_seq(r.entropy_seq, max_len)
            seqs.append(np.stack([e, d])); labels.append(1)
        for r in neg:
            e = pad_seq(r.entropy_seq, max_len)
            d = delta_seq(r.entropy_seq, max_len)
            seqs.append(np.stack([e, d])); labels.append(0)
        self.X = torch.tensor(np.stack(seqs), dtype=torch.float32)  # (N,2,L)
        self.y = torch.tensor(labels, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


class LSTMDataset(Dataset):
    """LSTM-format: (N, L, 1)."""
    def __init__(self, pos, neg, max_len):
        seqs, labels = [], []
        for r in pos:
            seqs.append(pad_seq(r.entropy_seq, max_len)); labels.append(1)
        for r in neg:
            seqs.append(pad_seq(r.entropy_seq, max_len)); labels.append(0)
        self.X = torch.tensor(np.stack(seqs), dtype=torch.float32).unsqueeze(2)  # (N,L,1)
        self.y = torch.tensor(labels, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


# ── Model definitions ──────────────────────────────────────────────────────────

class CNN1D(nn.Module):
    """Baseline 1D CNN."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        )
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))
    def forward(self, x): return self.head(self.net(x)).squeeze(1)


class MultiScaleCNN(nn.Module):
    """Parallel conv branches with kernel sizes 3, 5, 7."""
    def __init__(self, in_channels=1):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, 64, kernel_size=k, padding=k//2), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1), nn.Flatten()
            ) for k in [3, 5, 7]
        ])
        self.head = nn.Sequential(nn.Linear(64*3, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))
    def forward(self, x):
        return self.head(torch.cat([b(x) for b in self.branches], dim=1)).squeeze(1)


class TwoChannelCNN(nn.Module):
    """CNN with 2 input channels: entropy + delta."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        )
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))
    def forward(self, x): return self.head(self.net(x)).squeeze(1)


class LSTMClassifier(nn.Module):
    def __init__(self, hidden=64, layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0, bidirectional=False)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        out, (h, _) = self.lstm(x)
        return self.head(h[-1]).squeeze(1)


class GRUClassifier(nn.Module):
    def __init__(self, hidden=64, layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(1, hidden, layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        out, h = self.gru(x)
        return self.head(h[-1]).squeeze(1)


class CNNLSTMClassifier(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(32, hidden, 1, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 1))
    def forward(self, x):
        x = self.cnn(x).permute(0, 2, 1)  # (N, L/2, 32)
        _, (h, _) = self.lstm(x)
        return self.head(h[-1]).squeeze(1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)
    def forward(self, x): return x + self.pe[:, :x.size(1)]


class TransformerClassifier(nn.Module):
    def __init__(self, d_model=32, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pos_enc    = PositionalEncoding(d_model, MAX_LEN + 10)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=64,
                                               dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))
    def forward(self, x):
        # x: (N,1,L) → (N,L,1) → project → encode → avg pool
        x = x.permute(0, 2, 1)
        x = self.pos_enc(self.input_proj(x))
        x = self.encoder(x).mean(dim=1)
        return self.head(x).squeeze(1)


# ── Generic PyTorch trainer ────────────────────────────────────────────────────

def train_eval_torch(model, train_ds, test_ds, pos_n, neg_n, epochs, batch_size, device, lr=1e-3):
    model = model.to(device)
    pos_weight = torch.tensor([neg_n / pos_n], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1:3d}/{epochs}  loss={total_loss/len(train_loader):.4f}")

    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            p = (torch.sigmoid(model(X_batch.to(device))) > 0.5).cpu().numpy().astype(int)
            preds.extend(p); labels.extend(y_batch.numpy().astype(int))
    return np.array(labels), np.array(preds)


# ── Individual method runners ──────────────────────────────────────────────────

def run_sklearn(name, clf, X_train, y_train, X_test, y_test, config):
    section(name)
    print_config(config)
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    sklearn_cv(pipe, X_train, y_train)
    pipe.fit(X_train, y_train)
    return compute_and_print(y_test, pipe.predict(X_test))


def run_torch_seq(name, model, train_pos, train_neg, test_pos, test_neg,
                  config, epochs, batch_size, device, dataset_cls=SeqDataset):
    section(name)
    print_config(config)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters : {n_params:,}")
    train_ds = dataset_cls(train_pos, train_neg, MAX_LEN)
    test_ds  = dataset_cls(test_pos,  test_neg,  MAX_LEN)
    y_true, y_pred = train_eval_torch(model, train_ds, test_ds,
                                      len(train_pos), len(train_neg),
                                      epochs, batch_size, device)
    return compute_and_print(y_true, y_pred)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    a_results = load(args.results_dir, "A_pii")
    b_results = load(args.results_dir, "B_general")
    c_results = load(args.results_dir, "C_no_context")

    a_located     = [r for r in a_results if len(r.pii_token_positions) > 0]
    a_not_located = [r for r in a_results if len(r.pii_token_positions) == 0]

    loc_train,     loc_test     = split_samples(a_located,     args.test_ratio, args.seed)
    not_loc_train, not_loc_test = split_samples(a_not_located, args.test_ratio, args.seed)
    b_train,       b_test       = split_samples(b_results,     args.test_ratio, args.seed)
    c_test = c_results

    neg_train = not_loc_train + b_train
    neg_test  = not_loc_test  + b_test + c_test

    print(f"\n{'='*62}")
    print(f"  Entropy Sequence Classification — Method Comparison")
    print(f"{'='*62}")
    print(f"  Results dir : {args.results_dir}")
    print(f"  Split       : 80% train / 20% test, seed={args.seed}")
    print(f"  Train       : pos={len(loc_train)}, neg={len(neg_train)}")
    print(f"  Test        : pos={len(loc_test)},  neg={len(neg_test)}")
    print(f"  Device      : {args.device}, epochs={args.epochs}, batch={args.batch_size}")

    # Feature matrices for sklearn methods
    X_train = np.stack([extract_features(r) for r in loc_train + neg_train])
    y_train = np.array([1]*len(loc_train) + [0]*len(neg_train))
    X_test  = np.stack([extract_features(r) for r in loc_test  + neg_test])
    y_test  = np.array([1]*len(loc_test)  + [0]*len(neg_test))

    results = {}

    # 1. Logistic Regression
    cfg = {"C": 1.0, "max_iter": 1000, "features": 12}
    results["1. LR"] = run_sklearn("1. Logistic Regression", LogisticRegression(max_iter=1000, C=1.0),
                                   X_train, y_train, X_test, y_test, cfg)

    # 2. Random Forest
    cfg = {"n_estimators": 200, "max_depth": "None", "min_samples_leaf": 1}
    results["2. Random Forest"] = run_sklearn("2. Random Forest",
                                              RandomForestClassifier(n_estimators=200, random_state=args.seed),
                                              X_train, y_train, X_test, y_test, cfg)

    # 3. XGBoost
    if HAS_XGB:
        cfg = {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1, "subsample": 0.8}
        results["3. XGBoost"] = run_sklearn("3. XGBoost",
                                            XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                                          subsample=0.8, eval_metric="logloss",
                                                          random_state=args.seed, verbosity=0),
                                            X_train, y_train, X_test, y_test, cfg)
    else:
        print("\n  [3. XGBoost] SKIPPED — pip install xgboost")

    # 4. LSTM
    cfg = {"hidden": 64, "layers": 2, "dropout": 0.3, "lr": 1e-3, "epochs": args.epochs}
    results["4. LSTM"] = run_torch_seq("4. LSTM", LSTMClassifier(hidden=64, layers=2),
                                       loc_train, neg_train, loc_test, neg_test,
                                       cfg, args.epochs, args.batch_size, args.device,
                                       dataset_cls=LSTMDataset)

    # 5. GRU
    cfg = {"hidden": 64, "layers": 2, "dropout": 0.3, "lr": 1e-3, "epochs": args.epochs}
    results["5. GRU"] = run_torch_seq("5. GRU", GRUClassifier(hidden=64, layers=2),
                                      loc_train, neg_train, loc_test, neg_test,
                                      cfg, args.epochs, args.batch_size, args.device,
                                      dataset_cls=LSTMDataset)

    # 6. 1D CNN (baseline)
    cfg = {"channels": "1→32→64→128", "kernels": "5,3,3", "pool": "max+adaptive", "dropout": 0.3}
    results["6. CNN-1D"] = run_torch_seq("6. 1D CNN (baseline)", CNN1D(),
                                         loc_train, neg_train, loc_test, neg_test,
                                         cfg, args.epochs, args.batch_size, args.device)

    # 7. Multi-scale CNN
    cfg = {"channels": 1, "kernels": "3,5,7 (parallel)", "out_per_branch": 64, "dropout": 0.3}
    results["7. MultiScale-CNN"] = run_torch_seq("7. Multi-scale CNN (kernels 3,5,7)", MultiScaleCNN(in_channels=1),
                                                 loc_train, neg_train, loc_test, neg_test,
                                                 cfg, args.epochs, args.batch_size, args.device)

    # 8. 2-channel CNN (entropy + delta)
    cfg = {"channels": "2 (entropy+delta)", "kernels": "5,3,3", "pool": "max+adaptive", "dropout": 0.3}
    results["8. CNN-2ch"] = run_torch_seq("8. 2-channel CNN (entropy + Δentropy)", TwoChannelCNN(),
                                          loc_train, neg_train, loc_test, neg_test,
                                          cfg, args.epochs, args.batch_size, args.device,
                                          dataset_cls=TwoChannelDataset)

    # 9. CNN + LSTM
    cfg = {"cnn": "Conv1d(1→32,k=5)+MaxPool", "lstm": "hidden=64,layers=1", "dropout": 0.3}
    results["9. CNN+LSTM"] = run_torch_seq("9. CNN + LSTM", CNNLSTMClassifier(hidden=64),
                                           loc_train, neg_train, loc_test, neg_test,
                                           cfg, args.epochs, args.batch_size, args.device)

    # 10. Transformer
    cfg = {"d_model": 32, "nhead": 4, "num_layers": 2, "dropout": 0.1, "pos_enc": "sinusoidal"}
    results["10. Transformer"] = run_torch_seq("10. Transformer Encoder", TransformerClassifier(),
                                               loc_train, neg_train, loc_test, neg_test,
                                               cfg, args.epochs, args.batch_size, args.device)

    # ── Final summary ────────────────────────────────────────────────────────
    print(f"\n{'='*75}")
    print(f"  FINAL SUMMARY — Entropy Sequence Classification")
    print(f"  Task: PII-generated text vs. everything else (entropy signal only)")
    print(f"  Test set — Positive: {len(loc_test)}, Negative: {len(neg_test)}, Total: {len(loc_test)+len(neg_test)}")
    print(f"{'='*75}")
    print(f"{'Method':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'TP':>6} {'FP':>6} {'FN':>6}")
    print(f"{'-'*75}")
    for name, r in results.items():
        print(f"{name:<22} {r['acc']:>9.4f} {r['prec']:>10.4f} {r['rec']:>8.4f} {r['f1']:>8.4f}"
              f" {r['TP']:>6} {r['FP']:>6} {r['FN']:>6}")
    print(f"{'='*75}")


if __name__ == "__main__":
    main()
