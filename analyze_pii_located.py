"""
Compare entropy patterns between:
  - A_located:     Condition A samples where PII tokens were actually generated
  - A_not_located: Condition A samples where PII was NOT generated despite context

Usage:
    python analyze_pii_located.py --results-dir results_llama8b_20260401_1429
"""

import os
import sys
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
    return p.parse_args()


def load_a(results_dir):
    path = os.path.join(results_dir, "A_pii.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def split_by_pii_located(a_results):
    located     = [r for r in a_results if len(r.pii_token_positions) > 0]
    not_located = [r for r in a_results if len(r.pii_token_positions) == 0]
    return located, not_located


def entropy_stats(results, label):
    mean_entropies = [np.mean(r.entropy_seq) for r in results if r.entropy_seq]
    min_entropies  = [np.min(r.entropy_seq)  for r in results if r.entropy_seq]
    m1_fire_rate   = np.mean([len(r.red_flag_indices) > 0 for r in results])
    m2_fire_rate   = np.mean([len(r.sustained_flag_indices) > 0 for r in results])

    # Mean entropy at PII positions (only for located group)
    pii_entropies = []
    for r in results:
        for pos in r.pii_token_positions:
            if pos < len(r.entropy_seq):
                pii_entropies.append(r.entropy_seq[pos])

    print(f"\n{'─'*50}")
    print(f"  {label}  (n={len(results)})")
    print(f"{'─'*50}")
    print(f"  Mean entropy (avg over generation) : {np.mean(mean_entropies):.4f} ± {np.std(mean_entropies):.4f}")
    print(f"  Min  entropy (lowest token)        : {np.mean(min_entropies):.4f} ± {np.std(min_entropies):.4f}")
    print(f"  M1 fire rate                       : {m1_fire_rate:.3f}")
    print(f"  M2 fire rate                       : {m2_fire_rate:.3f}")
    if pii_entropies:
        print(f"  Entropy AT PII token positions     : {np.mean(pii_entropies):.4f} ± {np.std(pii_entropies):.4f}")

    return {
        "label": label,
        "n": len(results),
        "mean_entropy": np.mean(mean_entropies),
        "mean_entropy_std": np.std(mean_entropies),
        "min_entropy": np.mean(min_entropies),
        "min_entropy_std": np.std(min_entropies),
        "m1_fire_rate": m1_fire_rate,
        "m2_fire_rate": m2_fire_rate,
        "pii_entropy": np.mean(pii_entropies) if pii_entropies else None,
    }


def plot_comparison(stats_located, stats_not_located, results_dir):
    labels = ["A (PII located)", "A (PII not located)"]
    colors = ["#c0392b", "#2980b9"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("Entropy Comparison: A PII Located vs Not Located", fontsize=13)

    # Mean entropy
    vals = [stats_located["mean_entropy"], stats_not_located["mean_entropy"]]
    errs = [stats_located["mean_entropy_std"], stats_not_located["mean_entropy_std"]]
    axes[0].bar(labels, vals, yerr=errs, color=colors, capsize=5)
    axes[0].set_title("Mean Entropy (per sample)")
    axes[0].set_ylabel("nats")

    # Min entropy
    vals = [stats_located["min_entropy"], stats_not_located["min_entropy"]]
    errs = [stats_located["min_entropy_std"], stats_not_located["min_entropy_std"]]
    axes[1].bar(labels, vals, yerr=errs, color=colors, capsize=5)
    axes[1].set_title("Min Entropy (lowest token)")
    axes[1].set_ylabel("nats")

    # Fire rates
    x = np.arange(2)
    w = 0.35
    m1 = [stats_located["m1_fire_rate"], stats_not_located["m1_fire_rate"]]
    m2 = [stats_located["m2_fire_rate"], stats_not_located["m2_fire_rate"]]
    axes[2].bar(x - w/2, m1, w, label="M1 ΔH-drop", color="#e67e22")
    axes[2].bar(x + w/2, m2, w, label="M2 sustained", color="#8e44ad")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(["PII located", "PII not located"])
    axes[2].set_title("Red Flag Fire Rate")
    axes[2].set_ylabel("rate")
    axes[2].legend()

    plt.tight_layout()
    out = os.path.join(results_dir, "analysis_pii_located_vs_not.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {out}")


def plot_entropy_distributions(located, not_located, results_dir):
    """Distribution of per-sample mean entropy."""
    loc_means  = [np.mean(r.entropy_seq) for r in located  if r.entropy_seq]
    nloc_means = [np.mean(r.entropy_seq) for r in not_located if r.entropy_seq]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(loc_means,  bins=40, alpha=0.6, color="#c0392b", label=f"PII located (n={len(loc_means)})")
    ax.hist(nloc_means, bins=40, alpha=0.6, color="#2980b9", label=f"PII not located (n={len(nloc_means)})")
    ax.set_xlabel("Mean entropy per sample (nats)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Mean Entropy: Located vs Not Located")
    ax.legend()
    plt.tight_layout()
    out = os.path.join(results_dir, "analysis_entropy_distribution.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {out}")


def main():
    args = parse_args()
    a_results = load_a(args.results_dir)

    print(f"\n{'='*50}")
    print(f"Condition A Analysis: PII Located vs Not Located")
    print(f"Results dir: {args.results_dir}")
    print(f"Total A samples: {len(a_results)}")

    located, not_located = split_by_pii_located(a_results)
    print(f"  PII located    : {len(located)}")
    print(f"  PII not located: {len(not_located)}")

    stats_loc  = entropy_stats(located,     "A — PII located")
    stats_nloc = entropy_stats(not_located, "A — PII NOT located")

    # Entropy at PII positions vs non-PII positions within located samples
    print(f"\n{'='*50}")
    print("Entropy at PII token positions vs surrounding tokens (located group):")
    pii_ents, nonpii_ents = [], []
    for r in located:
        pii_set = set(r.pii_token_positions)
        for i, h in enumerate(r.entropy_seq):
            if i in pii_set:
                pii_ents.append(h)
            else:
                nonpii_ents.append(h)
    print(f"  At PII tokens     : {np.mean(pii_ents):.4f} ± {np.std(pii_ents):.4f}  (n={len(pii_ents)})")
    print(f"  At non-PII tokens : {np.mean(nonpii_ents):.4f} ± {np.std(nonpii_ents):.4f}  (n={len(nonpii_ents)})")

    plot_comparison(stats_loc, stats_nloc, args.results_dir)
    plot_entropy_distributions(located, not_located, args.results_dir)


if __name__ == "__main__":
    main()
