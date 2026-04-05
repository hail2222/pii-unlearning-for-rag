"""
Full Experiment Suite — 4 Experiments with Proper Data Protocol

Experiments:
  1. Entropy CNN-2ch (full 80-token sequence)  — entropy signal only
  2. Prefix CNN (K=5 tokens, real-time)        — entropy signal only, real-time
  3. Linear Probe standalone                   — hidden state signal only
  4. Pipeline: Prefix CNN (K=5) → Probe        — entropy + hidden state

Data protocol (no leakage):
  - ONE train/test split, seed=42, defined once and reused for all experiments
  - C_no_context: NEVER used for training, evaluated separately at the end of each experiment
  - CNN hyperparameters: fixed architecture (best from k-fold CV in hyperparam_tuning_v2)
  - Probe C (regularisation): selected by k-fold CV on training hidden states only
  - StandardScaler: fit on training hidden states only, transform test/C separately

Evaluation structure per experiment:
  [A] Main test    — pos=A_located_test, neg=A_not_located_test + B_test
  [B] C_no_context — neg=C_no_context (all), pos=none  → reports FP rate only
  [C] Combined     — A+B combined (for comparison with prior results)

Usage:
    python run_all_experiments.py \\
        --results-dir results_llama8b_20260401_1429 \\
        --epochs 30 \\
        --device cuda \\
        2>&1 | tee results_llama8b_20260401_1429/all_experiments.log
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

from run_experiment import SampleResult
from config import RESULTS_DIR as _DEFAULT_RESULTS_DIR, MAX_NEW_TOKENS

MAX_LEN = MAX_NEW_TOKENS  # 80
PREFIX_K = 5              # real-time detection prefix length


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
                   help="Folds for probe C selection (train set only)")
    # CNN-2ch architecture — defaults to best config from hyperparam_tuning_v2
    p.add_argument("--cnn-channels", type=int,  nargs=3, default=[64, 128, 256])
    p.add_argument("--cnn-kernels",  type=int,  nargs=3, default=[5, 3, 3])
    p.add_argument("--cnn-dropout",  type=float, default=0.3)
    return p.parse_args()


# ── Data utils ─────────────────────────────────────────────────────────────────

def load_pkl(results_dir, condition):
    with open(os.path.join(results_dir, f"{condition}.pkl"), "rb") as f:
        return pickle.load(f)


def split_samples(samples, test_ratio, seed):
    """Reproducible sample-level train/test split."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(samples))
    n_test = max(1, int(len(samples) * test_ratio))
    test_idx = set(idx[:n_test])
    train = [s for i, s in enumerate(samples) if i not in test_idx]
    test  = [s for i, s in enumerate(samples) if i in test_idx]
    return train, test


def pad_seq(seq, max_len):
    seq = np.array(seq, dtype=np.float32)
    return seq[:max_len] if len(seq) >= max_len else np.pad(seq, (0, max_len - len(seq)))


def delta_seq(seq, max_len):
    s = np.array(seq, dtype=np.float32)
    d = np.abs(np.diff(s, prepend=s[0] if len(s) > 0 else 0.0))
    return pad_seq(d, max_len)


def pad_prefix(seq, k):
    seq = np.array(seq, dtype=np.float32)[:k]
    return np.pad(seq, (0, k - len(seq))) if len(seq) < k else seq


# ── Datasets ───────────────────────────────────────────────────────────────────

class TwoChannelDataset(Dataset):
    """entropy + Δentropy, shape (N, 2, MAX_LEN)."""
    def __init__(self, pos, neg):
        data, labels = [], []
        for r in pos:
            data.append(np.stack([pad_seq(r.entropy_seq, MAX_LEN),
                                   delta_seq(r.entropy_seq, MAX_LEN)])); labels.append(1)
        for r in neg:
            data.append(np.stack([pad_seq(r.entropy_seq, MAX_LEN),
                                   delta_seq(r.entropy_seq, MAX_LEN)])); labels.append(0)
        self.X = torch.tensor(np.stack(data), dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


class PrefixDataset(Dataset):
    """entropy[0:PREFIX_K], shape (N, 1, PREFIX_K)."""
    def __init__(self, pos, neg):
        data, labels = [], []
        for r in pos:
            data.append(pad_prefix(r.entropy_seq, PREFIX_K)); labels.append(1)
        for r in neg:
            data.append(pad_prefix(r.entropy_seq, PREFIX_K)); labels.append(0)
        self.X = torch.tensor(np.stack(data), dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(labels, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


# ── Models ─────────────────────────────────────────────────────────────────────

class TwoChannelCNN(nn.Module):
    def __init__(self, channels=(64, 128, 256), kernels=(5, 3, 3), dropout=0.3):
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


class PrefixCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32,  kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1)
        )
    def forward(self, x): return self.head(self.net(x)).squeeze(1)


# ── CNN trainer ────────────────────────────────────────────────────────────────

def train_cnn(model, train_ds, epochs, batch_size, device, verbose=True):
    pos_n = int(train_ds.y.sum().item())
    neg_n = len(train_ds) - pos_n
    pos_weight = torch.tensor([neg_n / pos_n], dtype=torch.float32).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = model.to(device)
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
        if verbose and (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1:3d}/{epochs}  loss={total/len(loader):.4f}")
    return model


def predict_cnn(model, ds, device):
    loader = DataLoader(ds, batch_size=64)
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for X, y in loader:
            p = (torch.sigmoid(model(X.to(device))) > 0.5).cpu().numpy().astype(int)
            preds.extend(p.tolist())
            labels.extend(y.numpy().astype(int).tolist())
    return np.array(labels), np.array(preds)


# ── Probe builder ──────────────────────────────────────────────────────────────

def build_probe(train_pos, train_neg, cv_folds):
    """
    Train LR probe on red_flag_hidden_states from training split ONLY.
    C is selected by cross-validation on training data.
    Returns: fitted probe pipeline (StandardScaler + LR)
    """
    X_list, y_list = [], []
    for r in train_pos:
        for hs in r.red_flag_hidden_states:
            X_list.append(hs); y_list.append(1)
    for r in train_neg:
        for hs in r.red_flag_hidden_states:
            X_list.append(hs); y_list.append(0)

    if not X_list:
        raise ValueError("No red_flag_hidden_states in training data.")

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list)
    print(f"  Probe train: PII_hs={y.sum()}, NonPII_hs={(y==0).sum()}, dim={X.shape[1]}")

    # Select C by CV on training data (test set never touched)
    best_C, best_cv_f1 = 1.0, -1.0
    k = min(cv_folds, int(y.sum()), int((y == 0).sum()))
    if k >= 2:
        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
            pipe = SkPipeline([("sc", StandardScaler()),
                               ("lr", LogisticRegression(C=C, max_iter=1000))])
            scores = cross_val_score(pipe, X, y, cv=cv, scoring="f1")
            print(f"    C={C:<6}  CV F1={scores.mean():.4f} ± {scores.std():.4f}")
            if scores.mean() > best_cv_f1:
                best_cv_f1, best_C = scores.mean(), C
        print(f"  → Best C={best_C}  (CV F1={best_cv_f1:.4f})")
    else:
        print(f"  → Using C=1.0 (not enough samples for CV)")

    probe = SkPipeline([("sc", StandardScaler()),
                        ("lr", LogisticRegression(C=best_C, max_iter=1000))])
    probe.fit(X, y)
    return probe


def probe_predict_sample(sample, probe, fallback_positive=False):
    """Run probe on one sample's red_flag_hidden_states. Returns bool."""
    hs = sample.red_flag_hidden_states
    if not hs:
        return fallback_positive
    X = np.stack(hs).astype(np.float32)
    return bool(probe.predict(X).any())


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred):
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    acc  = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
    return {"f1": f1, "prec": prec, "rec": rec, "acc": acc,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn}


def print_metrics(m, label="", n_pos=None, n_neg=None):
    tag = f"[{label}] " if label else ""
    print(f"  {tag}F1={m['f1']:.4f}  Prec={m['prec']:.4f}  "
          f"Rec={m['rec']:.4f}  Acc={m['acc']:.4f}")
    print(f"    TP={m['TP']}  FP={m['FP']}  FN={m['FN']}  TN={m['TN']}"
          + (f"  (pos={n_pos}, neg={n_neg})" if n_pos else ""))


def eval_section(tag, y_true, y_pred, n_pos=None, n_neg=None):
    m = compute_metrics(y_true, y_pred)
    print_metrics(m, tag, n_pos, n_neg)
    return m


def header(title):
    print(f"\n{'='*68}")
    print(f"  {title}")
    print(f"{'='*68}")


# ── Experiment 1: Entropy CNN-2ch ──────────────────────────────────────────────

def exp1_entropy_cnn(train_pos, train_neg, test_pos, test_neg, c_neg, args):
    header("Experiment 1: Entropy CNN-2ch (full 80-token sequence)")
    print(f"  Model: TwoChannelCNN (entropy + Δentropy)")
    print(f"  Architecture: channels={tuple(args.cnn_channels)}, "
          f"kernels={tuple(args.cnn_kernels)}, dropout={args.cnn_dropout}")
    print(f"  Params: {sum(p.numel() for p in TwoChannelCNN(args.cnn_channels, args.cnn_kernels, args.cnn_dropout).parameters()):,}")
    print(f"  Data leakage check:")
    print(f"    - No feature scaling (CNN processes raw sequences)")
    print(f"    - Architecture is fixed (from hyperparam_tuning_v2 best config)")
    print(f"    - C_no_context excluded from training")

    model = TwoChannelCNN(tuple(args.cnn_channels),
                          tuple(args.cnn_kernels),
                          args.cnn_dropout)
    train_ds = TwoChannelDataset(train_pos, train_neg)
    model = train_cnn(model, train_ds, args.epochs, args.batch_size, args.device)

    # [A] Main test (A_not_located + B as negatives)
    test_ds_main = TwoChannelDataset(test_pos, test_neg)
    y_true_m, y_pred_m = predict_cnn(model, test_ds_main, args.device)
    m_main = eval_section("A: Main test", y_true_m, y_pred_m,
                          n_pos=len(test_pos), n_neg=len(test_neg))

    # [B] C_no_context only (all negative, measure FP rate)
    y_true_c = np.zeros(len(c_neg), dtype=int)
    c_ds = TwoChannelDataset([], c_neg)
    _, y_pred_c_raw = predict_cnn(model, c_ds, args.device)
    # predict_cnn returns (labels, preds); for empty pos, labels=all 0
    c_fp = int(y_pred_c_raw.sum())
    c_tn = len(c_neg) - c_fp
    print(f"  [B: C_no_context] FP={c_fp}/{len(c_neg)}  "
          f"FP_rate={c_fp/len(c_neg)*100:.1f}%  TN={c_tn}")

    # [C] Combined (main + C)
    y_true_all = np.concatenate([y_true_m, np.zeros(len(c_neg), dtype=int)])
    y_pred_all = np.concatenate([y_pred_m, y_pred_c_raw])
    m_comb = eval_section("C: Combined (main + C)", y_true_all, y_pred_all,
                          n_pos=len(test_pos), n_neg=len(test_neg) + len(c_neg))

    return {"main": m_main, "c_fp": c_fp, "c_total": len(c_neg), "combined": m_comb}


# ── Experiment 2: Prefix CNN (K=5) ────────────────────────────────────────────

def exp2_prefix_cnn(train_pos, train_neg, test_pos, test_neg, c_neg, args):
    header(f"Experiment 2: Prefix CNN (K={PREFIX_K}, real-time)")
    print(f"  Model: PrefixCNN — sees only entropy[0:{PREFIX_K}]")
    print(f"  Params: {sum(p.numel() for p in PrefixCNN().parameters()):,}")
    print(f"  Latency: median PII at token 9 → detected at token {PREFIX_K} → +{9-PREFIX_K} tokens lead")
    print(f"  Data leakage check:")
    print(f"    - Input is entropy[0:{PREFIX_K}] only (no future information)")
    print(f"    - Fixed architecture (no test-set selection)")
    print(f"    - C_no_context excluded from training")

    model = PrefixCNN()
    train_ds = PrefixDataset(train_pos, train_neg)
    model = train_cnn(model, train_ds, args.epochs, args.batch_size, args.device)

    # [A] Main test
    test_ds_main = PrefixDataset(test_pos, test_neg)
    y_true_m, y_pred_m = predict_cnn(model, test_ds_main, args.device)
    m_main = eval_section("A: Main test", y_true_m, y_pred_m,
                          n_pos=len(test_pos), n_neg=len(test_neg))

    # [B] C_no_context only
    c_ds = PrefixDataset([], c_neg)
    _, y_pred_c = predict_cnn(model, c_ds, args.device)
    c_fp = int(y_pred_c.sum())
    c_tn = len(c_neg) - c_fp
    print(f"  [B: C_no_context] FP={c_fp}/{len(c_neg)}  "
          f"FP_rate={c_fp/len(c_neg)*100:.1f}%  TN={c_tn}")

    # [C] Combined
    y_true_all = np.concatenate([y_true_m, np.zeros(len(c_neg), dtype=int)])
    y_pred_all = np.concatenate([y_pred_m, y_pred_c])
    m_comb = eval_section("C: Combined (main + C)", y_true_all, y_pred_all,
                          n_pos=len(test_pos), n_neg=len(test_neg) + len(c_neg))

    return {"main": m_main, "c_fp": c_fp, "c_total": len(c_neg),
            "combined": m_comb, "model": model}


# ── Experiment 3: Linear Probe standalone ─────────────────────────────────────

def exp3_linear_probe(train_pos, train_neg, test_pos, test_neg, c_neg, args):
    header("Experiment 3: Linear Probe standalone (hidden states only)")
    print(f"  Signal: red_flag_hidden_states (ΔH-drop moments, LLaMA layer 24, dim=4096)")
    print(f"  Data leakage check:")
    print(f"    - StandardScaler fit on TRAINING hidden states only")
    print(f"    - Probe C selected by {args.cv_folds}-fold CV on training data only")
    print(f"    - Hidden states are pre-computed from LLaMA (not learned)")
    print(f"    - C_no_context excluded from training")
    print()

    # Build probe using train split only
    probe = build_probe(train_pos, train_neg, args.cv_folds)

    # Coverage stats
    pos_no_hs = sum(1 for r in test_pos if not r.red_flag_hidden_states)
    neg_no_hs = sum(1 for r in test_neg if not r.red_flag_hidden_states)
    c_no_hs   = sum(1 for r in c_neg   if not r.red_flag_hidden_states)
    print(f"  Hidden state coverage:")
    print(f"    test_pos: {len(test_pos)-pos_no_hs}/{len(test_pos)} have hs "
          f"({(len(test_pos)-pos_no_hs)/len(test_pos)*100:.1f}%)")
    print(f"    test_neg: {len(test_neg)-neg_no_hs}/{len(test_neg)} have hs "
          f"({(len(test_neg)-neg_no_hs)/len(test_neg)*100:.1f}%)")
    print(f"    c_neg:    {len(c_neg)-c_no_hs}/{len(c_neg)} have hs "
          f"({(len(c_neg)-c_no_hs)/len(c_neg)*100:.1f}%)")
    print(f"  Fallback (no hidden states): predict NEGATIVE")

    # [A] Main test
    y_true_m = np.array([1]*len(test_pos) + [0]*len(test_neg))
    y_pred_m = np.array(
        [int(probe_predict_sample(r, probe, fallback_positive=False)) for r in test_pos] +
        [int(probe_predict_sample(r, probe, fallback_positive=False)) for r in test_neg]
    )
    m_main = eval_section("A: Main test", y_true_m, y_pred_m,
                          n_pos=len(test_pos), n_neg=len(test_neg))

    # [B] C_no_context only
    y_pred_c = np.array([int(probe_predict_sample(r, probe, fallback_positive=False))
                         for r in c_neg])
    c_fp = int(y_pred_c.sum())
    c_tn = len(c_neg) - c_fp
    print(f"  [B: C_no_context] FP={c_fp}/{len(c_neg)}  "
          f"FP_rate={c_fp/len(c_neg)*100:.1f}%  TN={c_tn}")

    # [C] Combined
    y_true_all = np.concatenate([y_true_m, np.zeros(len(c_neg), dtype=int)])
    y_pred_all = np.concatenate([y_pred_m, y_pred_c])
    m_comb = eval_section("C: Combined (main + C)", y_true_all, y_pred_all,
                          n_pos=len(test_pos), n_neg=len(test_neg) + len(c_neg))

    return {"main": m_main, "c_fp": c_fp, "c_total": len(c_neg),
            "combined": m_comb, "probe": probe}


# ── Experiment 4: Pipeline (Prefix CNN K=5 → Probe) ───────────────────────────

def exp4_pipeline(train_pos, train_neg, test_pos, test_neg, c_neg,
                  prefix_model, probe, args):
    header(f"Experiment 4: Pipeline — Prefix CNN (K={PREFIX_K}) → Linear Probe")
    print(f"  Step 1: Prefix CNN sees entropy[0:{PREFIX_K}] → flags samples as PII/non-PII")
    print(f"  Step 2: For CNN-positive samples, run Probe on hidden states")
    print(f"  Fallback (CNN=1 but no hidden states): predict POSITIVE (conservative)")
    print(f"  Data leakage check:")
    print(f"    - Prefix CNN trained on train set (reused from Exp 2)")
    print(f"    - Probe trained on train set (reused from Exp 3)")
    print(f"    - No new fitting on test data")
    print(f"    - C_no_context excluded from training")

    def pipeline_predict(samples):
        """Return predictions for a list of samples using CNN → Probe."""
        ds = PrefixDataset(samples, [])
        loader = DataLoader(ds, batch_size=64)
        prefix_model.eval()
        cnn_preds = []
        with torch.no_grad():
            for X, _ in loader:
                p = (torch.sigmoid(prefix_model(X.to(args.device))) > 0.5).cpu().numpy().astype(int)
                cnn_preds.extend(p.tolist())

        final_preds = []
        for i, r in enumerate(samples):
            if cnn_preds[i] == 0:
                final_preds.append(0)
            else:
                # CNN flagged → verify with probe
                final_preds.append(int(probe_predict_sample(r, probe, fallback_positive=True)))
        return np.array(final_preds)

    # [A] Main test
    pred_pos = pipeline_predict(test_pos)
    pred_neg = pipeline_predict(test_neg)
    y_true_m = np.array([1]*len(test_pos) + [0]*len(test_neg))
    y_pred_m = np.concatenate([pred_pos, pred_neg])
    m_main = eval_section("A: Main test", y_true_m, y_pred_m,
                          n_pos=len(test_pos), n_neg=len(test_neg))

    # FP breakdown for test_neg
    notloc_end = len(test_neg) - len([r for r in test_neg if r.condition == "B"])
    # Compute per-condition breakdown
    notloc_preds = pred_neg[[r.condition == "A" for r in test_neg]]
    b_preds      = pred_neg[[r.condition == "B" for r in test_neg]]
    notloc_fp = int(notloc_preds.sum())
    b_fp      = int(b_preds.sum())
    print(f"    FP breakdown — A_not_located={notloc_fp}  B_general={b_fp}")

    # [B] C_no_context
    pred_c = pipeline_predict(c_neg)
    c_fp   = int(pred_c.sum())
    c_tn   = len(c_neg) - c_fp
    print(f"  [B: C_no_context] FP={c_fp}/{len(c_neg)}  "
          f"FP_rate={c_fp/len(c_neg)*100:.1f}%  TN={c_tn}")

    # [C] Combined
    y_true_all = np.concatenate([y_true_m, np.zeros(len(c_neg), dtype=int)])
    y_pred_all = np.concatenate([y_pred_m, pred_c])
    m_comb = eval_section("C: Combined (main + C)", y_true_all, y_pred_all,
                          n_pos=len(test_pos), n_neg=len(test_neg) + len(c_neg))

    return {"main": m_main, "c_fp": c_fp, "c_total": len(c_neg), "combined": m_comb}


# ── Final summary printer ──────────────────────────────────────────────────────

def print_final_summary(results, n_pos, n_main_neg, n_c):
    header("FINAL SUMMARY")
    print(f"  Test set: pos={n_pos}  main_neg={n_main_neg}  C_no_context={n_c}")
    print()

    rows = [
        ("Exp1: CNN-2ch (80t)",         results["exp1"]["main"],     results["exp1"]["c_fp"]),
        ("Exp2: Prefix CNN (K=5)",       results["exp2"]["main"],     results["exp2"]["c_fp"]),
        ("Exp3: Linear Probe",           results["exp3"]["main"],     results["exp3"]["c_fp"]),
        ("Exp4: Prefix CNN→Probe",       results["exp4"]["main"],     results["exp4"]["c_fp"]),
    ]

    # Main test table
    print(f"  [A] Main test (pos={n_pos}, neg={n_main_neg}):")
    print(f"  {'Method':<30} {'F1':>6} {'Prec':>6} {'Rec':>6} {'Acc':>6} "
          f"{'TP':>4} {'FP':>4} {'FN':>4} {'TN':>5}")
    print(f"  {'-'*75}")
    for name, m, _ in rows:
        print(f"  {name:<30} {m['f1']:6.4f} {m['prec']:6.4f} {m['rec']:6.4f} "
              f"{m['acc']:6.4f} {m['TP']:4d} {m['FP']:4d} {m['FN']:4d} {m['TN']:5d}")

    # C_no_context FP table
    print()
    print(f"  [B] C_no_context FP rate (all {n_c} samples are negative):")
    print(f"  {'Method':<30} {'FP':>6} {'FP_rate':>8} {'TN':>6}")
    print(f"  {'-'*55}")
    for name, _, c_fp in rows:
        c_tn = n_c - c_fp
        print(f"  {name:<30} {c_fp:6d} {c_fp/n_c*100:7.1f}%  {c_tn:6d}")

    # Combined table
    print()
    print(f"  [C] Combined (pos={n_pos}, neg={n_main_neg+n_c}):")
    print(f"  {'Method':<30} {'F1':>6} {'Prec':>6} {'Rec':>6} {'Acc':>6}")
    print(f"  {'-'*55}")
    for idx, (name, _, _) in enumerate(rows):
        m = results[f"exp{idx+1}"]["combined"]
        print(f"  {name:<30} {m['f1']:6.4f} {m['prec']:6.4f} {m['rec']:6.4f} {m['acc']:6.4f}")

    print(f"\n  Notes:")
    print(f"  - All experiments use identical train/test split (seed=42)")
    print(f"  - Probe C selected by {results['cv_folds']}-fold CV on training hidden states")
    print(f"  - CNN architecture fixed (no test-set hyperparameter selection)")
    print(f"  - C_no_context never used in training for any experiment")
    print("=" * 68)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 68)
    print("  Full Experiment Suite — 4 Experiments, Proper Data Protocol")
    print(f"  Results dir : {args.results_dir}")
    print(f"  Split       : 80/20, seed={args.seed}")
    print(f"  Epochs      : {args.epochs}  Device: {args.device}")
    print(f"  CV folds    : {args.cv_folds} (probe C selection, train only)")
    print("=" * 68)

    # ── Load & split (DONE ONCE, reused by all experiments) ───────────────────
    print("\n  Loading data and creating train/test split...")
    a_results = load_pkl(args.results_dir, "A_pii")
    b_results = load_pkl(args.results_dir, "B_general")
    c_results = load_pkl(args.results_dir, "C_no_context")

    a_located     = [r for r in a_results if len(r.pii_token_positions) > 0]
    a_not_located = [r for r in a_results if len(r.pii_token_positions) == 0]

    loc_train,    loc_test    = split_samples(a_located,     args.test_ratio, args.seed)
    notloc_train, notloc_test = split_samples(a_not_located, args.test_ratio, args.seed)
    b_train,      b_test      = split_samples(b_results,     args.test_ratio, args.seed)

    train_pos = loc_train
    train_neg = notloc_train + b_train
    test_pos  = loc_test
    test_neg  = notloc_test + b_test
    c_neg     = c_results   # C_no_context: separate, never in training

    print(f"  Train — pos={len(train_pos)}, neg={len(train_neg)}")
    print(f"  Test  — pos={len(test_pos)},  neg={len(test_neg)} "
          f"(not_loc={len(notloc_test)}, B={len(b_test)})")
    print(f"  C_no_context — {len(c_neg)} samples (additional evaluation only)")

    # ── Run experiments ────────────────────────────────────────────────────────
    results = {"cv_folds": args.cv_folds}

    results["exp1"] = exp1_entropy_cnn(
        train_pos, train_neg, test_pos, test_neg, c_neg, args)

    results["exp2"] = exp2_prefix_cnn(
        train_pos, train_neg, test_pos, test_neg, c_neg, args)

    results["exp3"] = exp3_linear_probe(
        train_pos, train_neg, test_pos, test_neg, c_neg, args)

    # Exp4 reuses the trained prefix CNN from Exp2 and probe from Exp3
    results["exp4"] = exp4_pipeline(
        train_pos, train_neg, test_pos, test_neg, c_neg,
        prefix_model=results["exp2"]["model"],
        probe=results["exp3"]["probe"],
        args=args,
    )

    # ── Final summary ──────────────────────────────────────────────────────────
    print_final_summary(results, len(test_pos), len(test_neg), len(c_neg))


if __name__ == "__main__":
    main()
