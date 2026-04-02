"""
Compare token-level entropy curves between:
  - A_located    : Condition A, PII was generated (context given + PII appears in output)
  - A_not_located: Condition A, PII was NOT generated (context given, but PII absent in output)

Both groups have the same experimental condition (RAG + PII context).
The only difference is whether PII actually appeared in the generated text.

Usage:
    python analyze_entropy_curve.py --results-dir results_llama8b_20260401_1429
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_experiment import SampleResult
from config import RESULTS_DIR as _DEFAULT_RESULTS_DIR


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=str, default=_DEFAULT_RESULTS_DIR)
    p.add_argument("--max-len", type=int, default=80,
                   help="Max token positions to plot (truncate/pad to this length)")
    return p.parse_args()


def load_a(results_dir):
    with open(os.path.join(results_dir, "A_pii.pkl"), "rb") as f:
        return pickle.load(f)


def mean_curve(results, max_len):
    """Compute mean ± std entropy curve across samples, aligned to max_len."""
    matrix = []
    for r in results:
        seq = np.array(r.entropy_seq[:max_len], dtype=np.float32)
        if len(seq) < max_len:
            seq = np.pad(seq, (0, max_len - len(seq)), constant_values=np.nan)
        matrix.append(seq)
    matrix = np.array(matrix)
    mean = np.nanmean(matrix, axis=0)
    std  = np.nanstd(matrix, axis=0)
    return mean, std


def plot_curves(located, not_located, results_dir, max_len):
    mean_loc,  std_loc  = mean_curve(located,     max_len)
    mean_nloc, std_nloc = mean_curve(not_located, max_len)
    x = np.arange(max_len)

    fig, ax = plt.subplots(figsize=(12, 5))

    # PII generated
    ax.plot(x, mean_loc,  color="#c0392b", linewidth=2,
            label=f"PII generated (n={len(located)})")
    ax.fill_between(x,
                    mean_loc  - std_loc,
                    mean_loc  + std_loc,
                    color="#c0392b", alpha=0.15)

    # PII not generated
    ax.plot(x, mean_nloc, color="#2980b9", linewidth=2,
            label=f"PII not generated (n={len(not_located)})")
    ax.fill_between(x,
                    mean_nloc - std_nloc,
                    mean_nloc + std_nloc,
                    color="#2980b9", alpha=0.15)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1,
               label="Absolute threshold (0.5 nats)")

    ax.set_xlabel("Token position", fontsize=13)
    ax.set_ylabel("Entropy (nats)", fontsize=13)
    ax.set_title("Token-level Entropy: PII Generated vs Not Generated\n"
                 "(Condition A — both groups have RAG + PII context)",
                 fontsize=13)
    ax.legend(fontsize=12)
    ax.set_xlim(0, max_len - 1)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    out = os.path.join(results_dir, "entropy_curve_located_vs_not.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close()


def plot_boxplot(located, not_located, results_dir):
    """Box plot of per-sample mean entropy."""
    loc_means  = [np.mean(r.entropy_seq) for r in located  if r.entropy_seq]
    nloc_means = [np.mean(r.entropy_seq) for r in not_located if r.entropy_seq]

    fig, ax = plt.subplots(figsize=(6, 5))
    bp = ax.boxplot([loc_means, nloc_means],
                    labels=["PII generated", "PII not generated"],
                    patch_artist=True,
                    medianprops=dict(color="white", linewidth=2))
    bp["boxes"][0].set_facecolor("#c0392b")
    bp["boxes"][1].set_facecolor("#2980b9")

    ax.set_ylabel("Mean entropy per sample (nats)", fontsize=12)
    ax.set_title("Distribution of Mean Entropy\n(Condition A, same RAG context)", fontsize=12)
    plt.tight_layout()
    out = os.path.join(results_dir, "entropy_boxplot_located_vs_not.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close()


def print_summary(located, not_located):
    def stats(results, label):
        means = [np.mean(r.entropy_seq) for r in results if r.entropy_seq]
        print(f"\n  {label} (n={len(results)})")
        print(f"    Mean entropy : {np.mean(means):.4f} ± {np.std(means):.4f}")
        print(f"    Median       : {np.median(means):.4f}")

    print("\n" + "="*50)
    print("Entropy Summary (Condition A only)")
    print("="*50)
    stats(located,     "PII generated    (context → PII in output)")
    stats(not_located, "PII not generated (context → no PII in output)")


def main():
    args = parse_args()
    a_results   = load_a(args.results_dir)
    located     = [r for r in a_results if len(r.pii_token_positions) > 0]
    not_located = [r for r in a_results if len(r.pii_token_positions) == 0]

    print_summary(located, not_located)
    plot_curves(located, not_located, args.results_dir, args.max_len)
    plot_boxplot(located, not_located, args.results_dir)


if __name__ == "__main__":
    main()
