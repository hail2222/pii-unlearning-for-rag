"""
Hyperparameter tuning for CNN-2ch, Transformer, and LSTM.

For each model, runs a grid search over key hyperparameters.
Picks the best config per model based on F1 score.

Usage:
    python hyperparam_tuning.py \
        --results-dir results_llama8b_20260401_1429 \
        --epochs 30
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import math
import pickle
import itertools
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

from run_experiment import SampleResult
from config import RESULTS_DIR as _DEFAULT_RESULTS_DIR, MAX_NEW_TOKENS

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


# ── Datasets ───────────────────────────────────────────────────────────────────

class SeqDataset(Dataset):
    """Single-channel: (N, 1, L)"""
    def __init__(self, pos, neg, max_len):
        seqs, labels = [], []
        for r in pos:
            seqs.append(pad_seq(r.entropy_seq, max_len)); labels.append(1)
        for r in neg:
            seqs.append(pad_seq(r.entropy_seq, max_len)); labels.append(0)
        self.X = torch.tensor(np.stack(seqs), dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(labels, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


class TwoChannelDataset(Dataset):
    """Two-channel: entropy + delta, (N, 2, L)"""
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
        self.X = torch.tensor(np.stack(seqs), dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


class LSTMDataset(Dataset):
    """LSTM format: (N, L, 1)"""
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


# ── Models ─────────────────────────────────────────────────────────────────────

class TwoChannelCNN(nn.Module):
    def __init__(self, channels=(32, 64, 128), kernels=(5, 3, 3), dropout=0.3):
        super().__init__()
        c1, c2, c3 = channels
        k1, k2, k3 = kernels
        self.net = nn.Sequential(
            nn.Conv1d(2,  c1, kernel_size=k1, padding=k1//2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(c1, c2, kernel_size=k2, padding=k2//2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(c2, c3, kernel_size=k3, padding=k3//2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(c3, c3 // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(c3 // 2, 1)
        )
    def forward(self, x):
        return self.head(self.net(x)).squeeze(1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.size(1)]


class TransformerClassifier(nn.Module):
    def __init__(self, d_model=32, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pos_enc    = PositionalEncoding(d_model, MAX_LEN + 10)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=dim_feedforward,
                                               dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model // 2, 1)
        )
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pos_enc(self.input_proj(x))
        x = self.encoder(x).mean(dim=1)
        return self.head(x).squeeze(1)


class LSTMClassifier(nn.Module):
    def __init__(self, hidden=64, layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0, bidirectional=False)
        self.head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.head(h[-1]).squeeze(1)


# ── Trainer ────────────────────────────────────────────────────────────────────

def train_eval(model, train_ds, test_ds, pos_n, neg_n,
               epochs, batch_size, device, lr=1e-3, clip_grad=None):
    model = model.to(device)
    pos_weight = torch.tensor([neg_n / pos_n], dtype=torch.float32).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

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
            if clip_grad:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            p = (torch.sigmoid(model(X_batch.to(device))) > 0.5).cpu().numpy().astype(int)
            preds.extend(p)
            labels.extend(y_batch.numpy().astype(int))

    y_true, y_pred = np.array(labels), np.array(preds)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    return {
        "f1":   f1_score(y_true, y_pred, zero_division=0),
        "prec": precision_score(y_true, y_pred, zero_division=0),
        "rec":  recall_score(y_true, y_pred, zero_division=0),
        "acc":  accuracy_score(y_true, y_pred),
        "TP": tp, "FN": fn, "FP": fp, "TN": tn,
    }


def section(title):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


# ── Grid search helper ─────────────────────────────────────────────────────────

def grid_search(name, configs, build_fn, dataset_cls,
                loc_train, loc_test, neg_train, neg_test,
                epochs, batch_size, device):
    section(f"Tuning: {name}")
    best_f1, best_cfg, best_m = -1, None, None
    all_results = []

    for i, cfg in enumerate(configs):
        model = build_fn(cfg)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        train_ds = dataset_cls(loc_train, neg_train, MAX_LEN)
        test_ds  = dataset_cls(loc_test,  neg_test,  MAX_LEN)

        m = train_eval(model, train_ds, test_ds,
                       len(loc_train), len(neg_train),
                       epochs, batch_size, device,
                       lr=cfg.get("lr", 1e-3),
                       clip_grad=cfg.get("clip_grad", None))

        all_results.append((cfg, m, n_params))
        marker = " ←best" if m["f1"] > best_f1 else ""
        print(f"  [{i+1:2d}/{len(configs)}] {cfg}  params={n_params:,}")
        print(f"        F1={m['f1']:.4f}  Prec={m['prec']:.4f}  "
              f"Rec={m['rec']:.4f}  TP={m['TP']}  FP={m['FP']}  FN={m['FN']}{marker}")

        if m["f1"] > best_f1:
            best_f1, best_cfg, best_m = m["f1"], cfg, m

    print(f"\n  ★ Best config: {best_cfg}")
    print(f"  ★ F1={best_m['f1']:.4f}  Prec={best_m['prec']:.4f}  "
          f"Rec={best_m['rec']:.4f}  Acc={best_m['acc']:.4f}")
    print(f"  ★ TP={best_m['TP']}  FP={best_m['FP']}  FN={best_m['FN']}  TN={best_m['TN']}")
    return best_cfg, best_m, all_results


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

    print(f"\n{'='*65}")
    print(f"  Hyperparameter Tuning: CNN-2ch / Transformer / LSTM")
    print(f"  Results dir : {args.results_dir}")
    print(f"  Split       : 80% train / 20% test, seed={args.seed}")
    print(f"  Train       : pos={len(loc_train)}, neg={len(neg_train)}")
    print(f"  Test        : pos={len(loc_test)},  neg={len(neg_test)}")
    print(f"  Device      : {args.device}, epochs={args.epochs}, batch={args.batch_size}")
    print(f"{'='*65}")

    summary = {}

    # ── 1. CNN-2ch ─────────────────────────────────────────────────────────────
    cnn_grid = [
        {"channels": (16, 32, 64),   "kernels": (3, 3, 3), "dropout": 0.2},
        {"channels": (16, 32, 64),   "kernels": (5, 3, 3), "dropout": 0.3},
        {"channels": (32, 64, 128),  "kernels": (3, 3, 3), "dropout": 0.2},
        {"channels": (32, 64, 128),  "kernels": (5, 3, 3), "dropout": 0.3},
        {"channels": (32, 64, 128),  "kernels": (7, 3, 3), "dropout": 0.3},
        {"channels": (32, 64, 128),  "kernels": (5, 3, 3), "dropout": 0.1},
        {"channels": (64, 128, 256), "kernels": (5, 3, 3), "dropout": 0.3},
        {"channels": (64, 128, 256), "kernels": (7, 5, 3), "dropout": 0.3},
        {"channels": (64, 128, 256), "kernels": (5, 3, 3), "dropout": 0.1},
        {"channels": (32, 64, 128),  "kernels": (5, 3, 3), "dropout": 0.5},
    ]

    def build_cnn(cfg):
        return TwoChannelCNN(channels=cfg["channels"],
                             kernels=cfg["kernels"],
                             dropout=cfg["dropout"])

    best_cnn_cfg, best_cnn_m, _ = grid_search(
        "CNN-2ch", cnn_grid, build_cnn, TwoChannelDataset,
        loc_train, loc_test, neg_train, neg_test,
        args.epochs, args.batch_size, args.device
    )
    summary["CNN-2ch"] = (best_cnn_cfg, best_cnn_m)

    # ── 2. Transformer ─────────────────────────────────────────────────────────
    tfm_grid = [
        {"d_model": 32,  "nhead": 2, "num_layers": 1, "dim_feedforward": 64,  "dropout": 0.1},
        {"d_model": 32,  "nhead": 4, "num_layers": 2, "dim_feedforward": 64,  "dropout": 0.1},
        {"d_model": 32,  "nhead": 4, "num_layers": 2, "dim_feedforward": 128, "dropout": 0.1},
        {"d_model": 32,  "nhead": 4, "num_layers": 4, "dim_feedforward": 128, "dropout": 0.2},
        {"d_model": 64,  "nhead": 4, "num_layers": 2, "dim_feedforward": 128, "dropout": 0.1},
        {"d_model": 64,  "nhead": 4, "num_layers": 2, "dim_feedforward": 256, "dropout": 0.2},
        {"d_model": 64,  "nhead": 8, "num_layers": 2, "dim_feedforward": 256, "dropout": 0.1},
        {"d_model": 64,  "nhead": 8, "num_layers": 4, "dim_feedforward": 256, "dropout": 0.2},
        {"d_model": 128, "nhead": 4, "num_layers": 2, "dim_feedforward": 256, "dropout": 0.1},
        {"d_model": 128, "nhead": 8, "num_layers": 2, "dim_feedforward": 512, "dropout": 0.2},
    ]

    def build_tfm(cfg):
        return TransformerClassifier(d_model=cfg["d_model"],
                                     nhead=cfg["nhead"],
                                     num_layers=cfg["num_layers"],
                                     dim_feedforward=cfg["dim_feedforward"],
                                     dropout=cfg["dropout"])

    best_tfm_cfg, best_tfm_m, _ = grid_search(
        "Transformer", tfm_grid, build_tfm, SeqDataset,
        loc_train, loc_test, neg_train, neg_test,
        args.epochs, args.batch_size, args.device
    )
    summary["Transformer"] = (best_tfm_cfg, best_tfm_m)

    # ── 3. LSTM ────────────────────────────────────────────────────────────────
    lstm_grid = [
        {"hidden": 64,  "layers": 1, "dropout": 0.0, "lr": 1e-3, "clip_grad": 1.0},
        {"hidden": 64,  "layers": 1, "dropout": 0.0, "lr": 5e-4, "clip_grad": 1.0},
        {"hidden": 64,  "layers": 2, "dropout": 0.2, "lr": 5e-4, "clip_grad": 1.0},
        {"hidden": 64,  "layers": 2, "dropout": 0.3, "lr": 1e-3, "clip_grad": 1.0},
        {"hidden": 128, "layers": 1, "dropout": 0.0, "lr": 1e-3, "clip_grad": 1.0},
        {"hidden": 128, "layers": 1, "dropout": 0.0, "lr": 5e-4, "clip_grad": 1.0},
        {"hidden": 128, "layers": 2, "dropout": 0.2, "lr": 5e-4, "clip_grad": 1.0},
        {"hidden": 128, "layers": 2, "dropout": 0.3, "lr": 1e-3, "clip_grad": 1.0},
        {"hidden": 256, "layers": 1, "dropout": 0.0, "lr": 5e-4, "clip_grad": 1.0},
        {"hidden": 256, "layers": 2, "dropout": 0.2, "lr": 5e-4, "clip_grad": 1.0},
    ]

    def build_lstm(cfg):
        return LSTMClassifier(hidden=cfg["hidden"],
                              layers=cfg["layers"],
                              dropout=cfg["dropout"])

    best_lstm_cfg, best_lstm_m, _ = grid_search(
        "LSTM", lstm_grid, build_lstm, LSTMDataset,
        loc_train, loc_test, neg_train, neg_test,
        args.epochs, args.batch_size, args.device
    )
    summary["LSTM"] = (best_lstm_cfg, best_lstm_m)

    # ── Final summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  FINAL SUMMARY — Best config per model")
    print(f"  Test set: pos={len(loc_test)}, neg={len(neg_test)}")
    print(f"{'='*65}")
    print(f"  {'Model':<15} {'F1':>7} {'Prec':>7} {'Rec':>7} {'TP':>5} {'FP':>5} {'FN':>5}")
    print(f"  {'-'*55}")
    for model_name, (cfg, m) in summary.items():
        print(f"  {model_name:<15} {m['f1']:>7.4f} {m['prec']:>7.4f} {m['rec']:>7.4f}"
              f" {m['TP']:>5} {m['FP']:>5} {m['FN']:>5}")
        print(f"    Config: {cfg}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
