"""
Hyperparameter Tuning v2 — Proper train/validation/test protocol.

Fix from v1: in v1, best config was selected using test set F1 directly.
             That contaminates the test set and inflates reported performance.

Correct protocol (this file):
  1. For each config: evaluate via k-fold CV on TRAINING SET ONLY
  2. Select best config by mean CV F1  (test set never touched)
  3. Re-train best config on FULL training set
  4. Evaluate ONCE on test set → report this as the final result

Compute cost: 10 configs × k_folds = 10×5 = 50 training runs (per model)
              + 1 final training run per model
              = ~51 training runs per model (vs 10 in v1)

Usage:
    python hyperparam_tuning_v2.py \\
        --results-dir results_llama8b_20260401_1429 \\
        --epochs 30 \\
        --cv-folds 5 \\
        --device cuda \\
        2>&1 | tee results_llama8b_20260401_1429/hyperparam_tuning_v2.log
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import math
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import StratifiedKFold

from run_experiment import SampleResult
from config import RESULTS_DIR as _DEFAULT_RESULTS_DIR, MAX_NEW_TOKENS

MAX_LEN = MAX_NEW_TOKENS  # 80


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=str,   default=_DEFAULT_RESULTS_DIR)
    p.add_argument("--test-ratio",  type=float, default=0.2)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--epochs",      type=int,   default=30)
    p.add_argument("--batch-size",  type=int,   default=32)
    p.add_argument("--device",      type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--cv-folds",    type=int,   default=5,
                   help="Number of folds for cross-validation on training set")
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

class TwoChannelDataset(Dataset):
    """CNN-2ch: entropy + Δentropy, (N, 2, MAX_LEN)."""
    def __init__(self, pos, neg):
        seqs, labels = [], []
        for r in pos:
            e = pad_seq(r.entropy_seq, MAX_LEN)
            d = delta_seq(r.entropy_seq, MAX_LEN)
            seqs.append(np.stack([e, d])); labels.append(1)
        for r in neg:
            e = pad_seq(r.entropy_seq, MAX_LEN)
            d = delta_seq(r.entropy_seq, MAX_LEN)
            seqs.append(np.stack([e, d])); labels.append(0)
        self.X = torch.tensor(np.stack(seqs), dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.float32)

    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


class SeqDataset(Dataset):
    """Single-channel entropy sequence, (N, 1, MAX_LEN)."""
    def __init__(self, pos, neg):
        seqs, labels = [], []
        for r in pos:
            seqs.append(pad_seq(r.entropy_seq, MAX_LEN)); labels.append(1)
        for r in neg:
            seqs.append(pad_seq(r.entropy_seq, MAX_LEN)); labels.append(0)
        self.X = torch.tensor(np.stack(seqs), dtype=torch.float32).unsqueeze(1)
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
            nn.Conv1d(2,  c1, kernel_size=k1, padding=k1 // 2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(c1, c2, kernel_size=k2, padding=k2 // 2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(c2, c3, kernel_size=k3, padding=k3 // 2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(c3, c3 // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(c3 // 2, 1)
        )
    def forward(self, x): return self.head(self.net(x)).squeeze(1)


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
        self.pos_enc = PositionalEncoding(d_model, MAX_LEN + 10)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(d_model // 2, 1)
        )
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pos_enc(self.input_proj(x))
        x = self.encoder(x).mean(dim=1)
        return self.head(x).squeeze(1)


# ── Training helpers ───────────────────────────────────────────────────────────

def train_one(model, train_ds, pos_n, neg_n, epochs, batch_size, device, lr=1e-3):
    """Train model on a dataset. Returns trained model."""
    model = model.to(device)
    pos_weight = torch.tensor([neg_n / pos_n], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        total = 0.0
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            total += loss.item()
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f"      Epoch {epoch+1:3d}/{epochs}  loss={total/len(loader):.4f}")
    return model


def evaluate_ds(model, ds, device):
    """Return (y_true, y_pred) arrays for a dataset."""
    loader = DataLoader(ds, batch_size=64)
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for X, y in loader:
            p = (torch.sigmoid(model(X.to(device))) > 0.5).cpu().numpy().astype(int)
            preds.extend(p.tolist())
            labels.extend(y.numpy().astype(int).tolist())
    return np.array(labels), np.array(preds)


def metrics(y_true, y_pred):
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    return {
        "f1":   f1_score(y_true, y_pred, zero_division=0),
        "prec": precision_score(y_true, y_pred, zero_division=0),
        "rec":  recall_score(y_true, y_pred, zero_division=0),
        "acc":  accuracy_score(y_true, y_pred),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
    }


# ── Cross-validation grid search (TRAIN SET ONLY) ─────────────────────────────

def cv_grid_search(name, configs, build_fn, dataset_cls,
                   train_pos, train_neg,
                   k_folds, epochs, batch_size, device):
    """
    For each config, run k-fold CV on the training set.
    Select best config by mean CV F1.
    Test set is never touched here.

    Returns: best_cfg, cv_f1_scores_per_config
    """
    print(f"\n{'='*65}")
    print(f"  CV Grid Search: {name}  ({k_folds}-fold CV on train set)")
    print(f"  Train: pos={len(train_pos)}, neg={len(train_neg)}")
    print(f"{'='*65}")

    # Build full training dataset for indexing
    full_train_ds = dataset_cls(train_pos, train_neg)
    y_all = full_train_ds.y.numpy().astype(int)

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    folds = list(skf.split(np.zeros(len(full_train_ds)), y_all))

    best_mean_f1 = -1
    best_cfg = None
    all_cv_results = []

    for i, cfg in enumerate(configs):
        fold_f1s = []
        for fold_idx, (tr_idx, val_idx) in enumerate(folds):
            tr_ds  = Subset(full_train_ds, tr_idx)
            val_ds = Subset(full_train_ds, val_idx)

            # Count pos/neg in this fold's training split for pos_weight
            y_tr = y_all[tr_idx]
            pos_n = int(y_tr.sum())
            neg_n = int((y_tr == 0).sum())
            if pos_n == 0 or neg_n == 0:
                continue

            model = build_fn(cfg)
            model = train_one(model, tr_ds, pos_n, neg_n,
                              epochs, batch_size, device)
            y_true, y_pred = evaluate_ds(model, val_ds, device)
            fold_f1 = f1_score(y_true, y_pred, zero_division=0)
            fold_f1s.append(fold_f1)

        mean_f1 = np.mean(fold_f1s) if fold_f1s else 0.0
        std_f1  = np.std(fold_f1s)  if fold_f1s else 0.0
        all_cv_results.append((cfg, mean_f1, std_f1))

        marker = " ←best" if mean_f1 > best_mean_f1 else ""
        n_params = sum(p.numel() for p in build_fn(cfg).parameters())
        print(f"  [{i+1:2d}/{len(configs)}] {cfg}  params={n_params:,}")
        print(f"          CV F1 = {mean_f1:.4f} ± {std_f1:.4f}{marker}")

        if mean_f1 > best_mean_f1:
            best_mean_f1 = mean_f1
            best_cfg = cfg

    print(f"\n  ★ Best config (by CV F1={best_mean_f1:.4f}): {best_cfg}")
    return best_cfg, all_cv_results


def final_train_eval(name, cfg, build_fn, dataset_cls,
                     train_pos, train_neg, test_pos, test_neg,
                     epochs, batch_size, device):
    """
    Re-train best config on FULL training set.
    Evaluate ONCE on test set.
    This is the only time test set is used.
    """
    print(f"\n  Re-training best {name} config on full training set...")
    train_ds = dataset_cls(train_pos, train_neg)
    test_ds  = dataset_cls(test_pos,  test_neg)

    pos_n = len(train_pos)
    neg_n = len(train_neg)

    model = build_fn(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}")

    model = train_one(model, train_ds, pos_n, neg_n, epochs, batch_size, device)
    y_true, y_pred = evaluate_ds(model, test_ds, device)
    m = metrics(y_true, y_pred)

    tp, fp, fn, tn = m["TP"], m["FP"], m["FN"], m["TN"]
    print(f"\n  [TEST SET — {name} best config]")
    print(f"    F1={m['f1']:.4f}  Prec={m['prec']:.4f}  Rec={m['rec']:.4f}  Acc={m['acc']:.4f}")
    print(f"    TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    return m


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 65)
    print("  Hyperparameter Tuning v2 — Proper CV Protocol")
    print(f"  Results dir : {args.results_dir}")
    print(f"  Split       : 80% train / 20% test, seed={args.seed}")
    print(f"  CV folds    : {args.cv_folds}  (on train set only)")
    print(f"  Epochs      : {args.epochs}  Device: {args.device}")
    print("=" * 65)
    print()
    print("  Protocol:")
    print("    Step 1. For each config: k-fold CV on training set → CV F1")
    print("    Step 2. Select best config by CV F1  [test set not used]")
    print("    Step 3. Re-train best config on full training set")
    print("    Step 4. Evaluate ONCE on test set → final reported result")
    print("=" * 65)

    # ── Load & split ───────────────────────────────────────────────────────────
    a_results = load(args.results_dir, "A_pii")
    b_results = load(args.results_dir, "B_general")
    c_results = load(args.results_dir, "C_no_context")

    a_located     = [r for r in a_results if len(r.pii_token_positions) > 0]
    a_not_located = [r for r in a_results if len(r.pii_token_positions) == 0]

    loc_train,     loc_test     = split_samples(a_located,     args.test_ratio, args.seed)
    not_loc_train, not_loc_test = split_samples(a_not_located, args.test_ratio, args.seed)
    b_train,       b_test       = split_samples(b_results,     args.test_ratio, args.seed)
    c_test = c_results

    train_pos = loc_train
    train_neg = not_loc_train + b_train
    test_pos  = loc_test
    test_neg  = not_loc_test + b_test + c_test

    print(f"\n  Train — pos={len(train_pos)}, neg={len(train_neg)}")
    print(f"  Test  — pos={len(test_pos)},  neg={len(test_neg)}")

    summary = {}

    # ════════════════════════════════════════════════════════════════════════════
    #  CNN-2ch
    # ════════════════════════════════════════════════════════════════════════════
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

    # Step 1 & 2: CV on train set → select best config
    best_cnn_cfg, cnn_cv_results = cv_grid_search(
        "CNN-2ch", cnn_grid, build_cnn, TwoChannelDataset,
        train_pos, train_neg,
        args.cv_folds, args.epochs, args.batch_size, args.device
    )

    # Step 3 & 4: Re-train on full train, evaluate ONCE on test
    cnn_test_m = final_train_eval(
        "CNN-2ch", best_cnn_cfg, build_cnn, TwoChannelDataset,
        train_pos, train_neg, test_pos, test_neg,
        args.epochs, args.batch_size, args.device
    )
    summary["CNN-2ch"] = (best_cnn_cfg, cnn_test_m)

    # ════════════════════════════════════════════════════════════════════════════
    #  Transformer
    # ════════════════════════════════════════════════════════════════════════════
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
        return TransformerClassifier(
            d_model=cfg["d_model"], nhead=cfg["nhead"],
            num_layers=cfg["num_layers"], dim_feedforward=cfg["dim_feedforward"],
            dropout=cfg["dropout"]
        )

    best_tfm_cfg, tfm_cv_results = cv_grid_search(
        "Transformer", tfm_grid, build_tfm, SeqDataset,
        train_pos, train_neg,
        args.cv_folds, args.epochs, args.batch_size, args.device
    )

    tfm_test_m = final_train_eval(
        "Transformer", best_tfm_cfg, build_tfm, SeqDataset,
        train_pos, train_neg, test_pos, test_neg,
        args.epochs, args.batch_size, args.device
    )
    summary["Transformer"] = (best_tfm_cfg, tfm_test_m)

    # ── Final summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  FINAL SUMMARY — Best config per model (proper CV protocol)")
    print(f"  Config selected by: {args.cv_folds}-fold CV F1 on train set")
    print(f"  Reported performance: single evaluation on held-out test set")
    print(f"  Test set: pos={len(test_pos)}, neg={len(test_neg)}")
    print(f"{'='*65}")
    print(f"  {'Model':<15} {'F1':>7} {'Prec':>7} {'Rec':>7} {'Acc':>7} {'TP':>5} {'FP':>5} {'FN':>5}")
    print(f"  {'-'*60}")
    for model_name, (cfg, m) in summary.items():
        print(f"  {model_name:<15} {m['f1']:>7.4f} {m['prec']:>7.4f} {m['rec']:>7.4f}"
              f" {m['acc']:>7.4f} {m['TP']:>5} {m['FP']:>5} {m['FN']:>5}")
        print(f"    Config: {cfg}")

    print(f"\n  [Reference: v1 results (test-set selection bias)]")
    print(f"  CNN-2ch      F1=0.9513  Prec=0.9333  Rec=0.9699  Acc=0.9879")
    print(f"  Transformer  F1=0.9397  Prec=0.8984  Rec=0.9849  Acc=0.9846")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
