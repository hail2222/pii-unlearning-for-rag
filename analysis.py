"""
Visualization and analysis of experiment results.
Run after run_experiment.py.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import csv
import json
import pickle
import argparse
import numpy as np
from run_experiment import SampleResult  # needed for pickle deserialization
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from config import RESULTS_DIR as _DEFAULT_RESULTS_DIR

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=str, default=_DEFAULT_RESULTS_DIR)
    return p.parse_args()

# Set at module level so helper functions can use it; overridden in __main__
RESULTS_DIR = _DEFAULT_RESULTS_DIR
PLOT_DIR    = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

CONDITION_COLORS = {
    "A_pii":        "#e74c3c",   # red
    "B_general":    "#2ecc71",   # green
    "C_no_context": "#f39c12",   # orange
}
CONDITION_LABELS = {
    "A_pii":        "A: PII (RAG context)",
    "B_general":    "B: General Knowledge",
    "C_no_context": "C: PII (No context)",
}


def load_results(results_dir: str = RESULTS_DIR) -> dict:
    results = {}
    for cond in ["A_pii", "B_general", "C_no_context"]:
        pkl = os.path.join(results_dir, f"{cond}.pkl")
        if os.path.exists(pkl):
            with open(pkl, "rb") as f:
                results[cond] = pickle.load(f)
    return results


# ── Plot 1: Token-labeled entropy time series (all samples, per condition) ────

def _clean_token(tok: str) -> str:
    """Make token printable for x-axis label."""
    tok = tok.replace("\n", "↵").replace("\t", "→")
    if len(tok) > 6:
        tok = tok[:5] + "…"
    return tok


def plot_token_entropy_all(results: dict):
    """
    For each condition, draw all 20 samples stacked vertically in one PNG.
    X-axis shows actual token strings. Red flags and PII positions marked.
    """
    for cond, samples in results.items():
        color = CONDITION_COLORS.get(cond, "blue")
        n = len(samples)

        fig, axes = plt.subplots(
            n, 1,
            figsize=(28, 3.5 * n),
            squeeze=False,
        )
        fig.suptitle(f"{CONDITION_LABELS[cond]} — Entropy per Token (all {n} samples)",
                     fontsize=13, fontweight="bold", y=1.001)

        for i, r in enumerate(samples):
            ax = axes[i][0]
            seq = np.array(r.entropy_seq)
            t   = np.arange(len(seq))

            ax.plot(t, seq, color=color, linewidth=1.2)
            ax.fill_between(t, seq, alpha=0.15, color=color)

            # Red flag verticals
            for rf in r.red_flag_indices:
                ax.axvline(rf, color="red", alpha=0.6, linestyle="--", linewidth=1)

            # PII token verticals
            for pp in r.pii_token_positions:
                ax.axvline(pp, color="purple", alpha=0.8, linestyle=":", linewidth=1.5)

            # X-axis: token strings
            ax.set_xticks(t)
            ax.set_xticklabels(
                [_clean_token(tok) for tok in r.tokens],
                fontsize=5.5, rotation=60, ha="right", fontfamily="monospace",
            )
            ax.set_xlim(-0.5, len(seq) - 0.5)
            ax.set_ylabel("H (nats)", fontsize=7)

            subject_label = r.subject or f"sample {i}"
            lead_str = f"  lead={r.lead_times}" if r.lead_times else ""
            ax.set_title(
                f"[{i+1}] {subject_label} | pii={r.pii_types}{lead_str}",
                fontsize=7, loc="left",
            )

            # Legend only on first subplot
            if i == 0:
                ax.legend(handles=[
                    Line2D([0], [0], color=color,    label="H_t"),
                    Line2D([0], [0], color="red",    linestyle="--", label="Red Flag (ΔH drop)"),
                    Line2D([0], [0], color="purple", linestyle=":",  label="PII token start"),
                ], fontsize=7, loc="upper right")

        plt.tight_layout()
        path = os.path.join(PLOT_DIR, f"token_entropy_{cond}.png")
        plt.savefig(path, dpi=130, bbox_inches="tight")
        print(f"Saved: {path}")
        plt.close()


# ── Plot 2: Entropy distribution at PII vs non-PII tokens ─────────────────────

def plot_entropy_distributions(results: dict):
    """Box plot of entropy values at PII-generating tokens vs all other tokens."""
    fig, ax = plt.subplots(figsize=(8, 5))

    data, tick_labels = [], []
    for cond, samples in results.items():
        pii_entropies, other_entropies = [], []
        for r in samples:
            seq = np.array(r.entropy_seq)
            pii_set = set(r.pii_token_positions)
            for j, h in enumerate(seq):
                if j in pii_set:
                    pii_entropies.append(h)
                else:
                    other_entropies.append(h)

        if pii_entropies:
            data.append(pii_entropies)
            tick_labels.append(f"{cond}\n(PII tokens)")
        if other_entropies:
            data.append(other_entropies)
            tick_labels.append(f"{cond}\n(other tokens)")

    ax.boxplot(data, tick_labels=tick_labels, patch_artist=True)
    ax.set_ylabel("Entropy (nats)")
    ax.set_title("Entropy Distribution: PII tokens vs Others")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "entropy_distributions.png")
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()


# ── Plot 3: Lead-time histogram ────────────────────────────────────────────────

def plot_lead_time(results: dict):
    """Histogram of lead-time values for Condition A."""
    a_results = results.get("A_pii", [])
    all_lead = [lt for r in a_results for lt in r.lead_times]

    if not all_lead:
        print("No lead-time data for Condition A (no PII tokens matched or no red flags).")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(all_lead, bins=range(0, max(all_lead) + 2), color="#e74c3c", edgecolor="white")
    ax.axvline(np.mean(all_lead), color="black", linestyle="--", label=f"Mean={np.mean(all_lead):.1f}")
    ax.set_xlabel("Lead-time (tokens before PII)")
    ax.set_ylabel("Count")
    ax.set_title("Red Flag Lead-time (Condition A: PII RAG)")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "lead_time.png")
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()


# ── CSV export ─────────────────────────────────────────────────────────────────

def export_csv(results: dict):
    """
    Export subject / pii_types / question / generated_answer as CSV per condition.
    Question is extracted from prompt (last line before Answer:).
    """
    for cond, samples in results.items():
        path = os.path.join(RESULTS_DIR, f"{cond}_table.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["subject", "pii_types", "question", "generated_answer"])
            for r in samples:
                # Extract question from prompt: line starting with "Question:"
                question = ""
                for line in r.prompt.splitlines():
                    if line.startswith("Question:"):
                        question = line.replace("Question:", "").strip()
                        break
                writer.writerow([
                    r.subject or "",
                    ", ".join(r.pii_types),
                    question,
                    r.generated_text.strip(),
                ])
        print(f"Saved: {path}")


# ── Plot 4: Absolute-level & persistence analysis ──────────────────────────────

ABS_THRESHOLD = 0.5   # nats — "low entropy" absolute cutoff


def _abs_run_lengths(entropy_seq: np.ndarray, ceiling: float = ABS_THRESHOLD) -> list[int]:
    """Lengths of consecutive runs where H_t < ceiling (absolute threshold)."""
    runs, current = [], 0
    for h in entropy_seq:
        if h < ceiling:
            current += 1
        else:
            if current > 0:
                runs.append(current)
            current = 0
    if current > 0:
        runs.append(current)
    return runs


def _abs_recovery_time(entropy_seq: np.ndarray, flag_idx: int,
                       ceiling: float = ABS_THRESHOLD) -> int:
    """Tokens until H_t rises back above ceiling after flag_idx."""
    for t in range(flag_idx, len(entropy_seq)):
        if entropy_seq[t] >= ceiling:
            return t - flag_idx
    return len(entropy_seq) - flag_idx


def _autocorr_lag1(entropy_seq: np.ndarray) -> float:
    """Pearson autocorrelation at lag 1."""
    if len(entropy_seq) < 3:
        return float("nan")
    x = entropy_seq[:-1] - entropy_seq[:-1].mean()
    y = entropy_seq[1:]  - entropy_seq[1:].mean()
    denom = np.std(entropy_seq[:-1]) * np.std(entropy_seq[1:])
    if denom < 1e-9:
        return float("nan")
    return float(np.mean(x * y) / denom)


def compute_abs_stats(samples) -> dict:
    mean_entropies, low_token_ratios, abs_run_lens, abs_recoveries, autocorrs, max_entropies = \
        [], [], [], [], [], []
    for r in samples:
        seq = np.array(r.entropy_seq)
        mean_entropies.append(float(seq.mean()))
        max_entropies.append(float(seq.max()))
        low_token_ratios.append(float((seq < ABS_THRESHOLD).mean()))
        abs_run_lens.extend(_abs_run_lengths(seq))
        if r.red_flag_indices:
            abs_recoveries.append(_abs_recovery_time(seq, r.red_flag_indices[0]))
        ac = _autocorr_lag1(seq)
        if not np.isnan(ac):
            autocorrs.append(ac)
    return {
        "mean_entropy":    mean_entropies,
        "max_entropy":     max_entropies,
        "low_ratio":       low_token_ratios,
        "abs_run_lengths": abs_run_lens,
        "abs_recoveries":  abs_recoveries,
        "autocorrs":       autocorrs,
    }


def plot_persistence(results: dict):
    """Five-panel absolute-level & persistence comparison across A / B / C."""
    conds  = list(results.keys())
    stats  = {c: compute_abs_stats(results[c]) for c in conds}
    colors = [CONDITION_COLORS.get(c, "blue") for c in conds]
    labels = [CONDITION_LABELS.get(c, c) for c in conds]

    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    fig.suptitle(f"Entropy Level & Persistence (abs threshold={ABS_THRESHOLD} nats)",
                 fontsize=12, fontweight="bold")

    def _boxpanel(ax, key, ylabel, title):
        data = [stats[c][key] for c in conds]
        bp = ax.boxplot(data, patch_artist=True)
        for patch, col in zip(bp["boxes"], colors):
            patch.set_facecolor(col); patch.set_alpha(0.7)
        ax.set_xticks(range(1, len(conds) + 1))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title, fontsize=8)
        for i, d in enumerate(data):
            if d:
                ypos = max(d) * 1.03
                ax.text(i + 1, ypos, f"μ={np.mean(d):.2f}", ha="center", fontsize=7)

    # Panel 1: Mean entropy (절대 레벨)
    _boxpanel(axes[0], "mean_entropy", "Entropy (nats)",
              "Mean Entropy\n(lower = more confident overall)")

    # Panel 2: Max entropy (피크 높이)
    _boxpanel(axes[1], "max_entropy", "Entropy (nats)",
              "Max Entropy\n(higher = more uncertain peaks)")

    # Panel 3: Ratio of tokens below threshold (저엔트로피 비율)
    _boxpanel(axes[2], "low_ratio", f"Ratio (H < {ABS_THRESHOLD})",
              f"Low-entropy Token Ratio\n(H < {ABS_THRESHOLD} nats)")

    # Panel 4: Absolute run length (핵심: drop 후 지속성)
    _boxpanel(axes[3], "abs_run_lengths", "Run length (tokens)",
              f"Abs. Low-entropy Run Length\n(consecutive tokens with H < {ABS_THRESHOLD})")

    # Panel 5: Absolute recovery time
    _boxpanel(axes[4], "abs_recoveries", "Tokens until recovery",
              f"Abs. Recovery Time\n(tokens to return H ≥ {ABS_THRESHOLD} after drop)")

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "persistence_analysis.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()

    # ── Numeric summary ────────────────────────────────────────────────────────
    print(f"\n── Absolute-level & Persistence Stats (threshold={ABS_THRESHOLD}) ──")
    for c in conds:
        s = stats[c]
        print(f"\n{CONDITION_LABELS.get(c, c)}")
        print(f"  Mean entropy          : {np.mean(s['mean_entropy']):.4f}")
        print(f"  Max entropy           : {np.mean(s['max_entropy']):.4f}  "
              f"(max across samples: {max(s['max_entropy']):.2f})")
        print(f"  Low-entropy ratio     : {np.mean(s['low_ratio']):.4f}  "
              f"(tokens < {ABS_THRESHOLD} nats)")
        if s["abs_run_lengths"]:
            print(f"  Abs run length        : mean={np.mean(s['abs_run_lengths']):.2f}, "
                  f"max={max(s['abs_run_lengths'])}")
        else:
            print(f"  Abs run length        : no runs below {ABS_THRESHOLD}")
        if s["abs_recoveries"]:
            print(f"  Abs recovery time     : mean={np.mean(s['abs_recoveries']):.2f}, "
                  f"median={np.median(s['abs_recoveries']):.1f}")


# ── Summary stats ──────────────────────────────────────────────────────────────

def print_summary(results: dict):
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    for cond, samples in results.items():
        n = len(samples)
        n_flags    = sum(1 for r in samples if r.red_flag_indices)
        n_pii_hit  = sum(1 for r in samples if r.pii_token_positions)
        all_lead   = [lt for r in samples for lt in r.lead_times]
        mean_flags = np.mean([len(r.red_flag_indices) for r in samples])

        print(f"\n{CONDITION_LABELS.get(cond, cond)} ({n} samples)")
        print(f"  Samples with ≥1 red flag : {n_flags}/{n} ({100*n_flags/n:.0f}%)")
        print(f"  Avg red flags per sample  : {mean_flags:.1f}")
        print(f"  Samples with PII matched  : {n_pii_hit}/{n}")
        if all_lead:
            print(f"  Lead-time (tokens)        : mean={np.mean(all_lead):.1f}, "
                  f"median={np.median(all_lead):.1f}, "
                  f"min={min(all_lead)}, max={max(all_lead)}")


if __name__ == "__main__":
    args = _parse_args()
    # Override global paths if --results-dir was given
    RESULTS_DIR = args.results_dir
    PLOT_DIR    = os.path.join(RESULTS_DIR, "plots")
    os.makedirs(PLOT_DIR, exist_ok=True)

    results = load_results(RESULTS_DIR)
    if not results:
        print("No results found. Run run_experiment.py first.")
    else:
        print_summary(results)
        plot_token_entropy_all(results)
        plot_entropy_distributions(results)
        plot_lead_time(results)
        plot_persistence(results)
        export_csv(results)
        print("\nAll outputs saved to:", RESULTS_DIR)
