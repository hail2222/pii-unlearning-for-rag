"""
Token-level entropy computation and Red Flag detection.
"""

import numpy as np
import torch
import torch.nn.functional as F


def compute_entropy(logits: torch.Tensor) -> float:
    """
    Compute Shannon entropy (in nats) from a single-step logit vector.
    H = -Σ p_i * log(p_i)

    Args:
        logits: shape (vocab_size,) — raw logits for one generation step
    Returns:
        scalar entropy value
    """
    probs = F.softmax(logits.float(), dim=-1)
    # clamp to avoid log(0)
    log_probs = torch.log(probs.clamp(min=1e-12))
    entropy = -(probs * log_probs).sum().item()
    return entropy


def compute_entropy_sequence(all_logits: list[torch.Tensor]) -> np.ndarray:
    """
    Compute per-token entropy for a full generated sequence.

    Args:
        all_logits: list of (vocab_size,) tensors, one per generation step
    Returns:
        np.ndarray of shape (T,) with entropy values
    """
    return np.array([compute_entropy(l) for l in all_logits])


def compute_delta_entropy(entropy_seq: np.ndarray) -> np.ndarray:
    """
    First-order difference of entropy sequence: ΔH_t = H_t - H_{t-1}.
    First element is 0 (no previous step).
    """
    delta = np.diff(entropy_seq, prepend=entropy_seq[0])
    return delta


def detect_red_flags(
    delta_entropy: np.ndarray,
    threshold: float,
    zscore: float = 0.0,
) -> list[int]:
    """
    Return indices where a sharp entropy drop occurs.

    If zscore > 0, uses adaptive threshold: mean(ΔH) - zscore * std(ΔH).
    Falls back to fixed -threshold if std is near zero.

    Args:
        delta_entropy: ΔH sequence
        threshold:     fixed fallback threshold (nats)
        zscore:        if > 0, use adaptive threshold instead
    Returns:
        list of token indices where Red Flag fires
    """
    if zscore > 0 and delta_entropy.std() > 1e-6:
        cutoff = delta_entropy.mean() - zscore * delta_entropy.std()
    else:
        cutoff = -threshold
    return [int(i) for i, dh in enumerate(delta_entropy) if dh < cutoff]


def find_pii_token_positions(
    generated_ids: list[int],
    pii_values_ids: list[list[int]],
) -> list[int]:
    """
    Find positions where any individual PII value appears in generated_ids.
    Each element of pii_values_ids is a list of candidate token-id sequences
    for the same PII string (with/without leading space variants).

    Returns sorted, deduplicated list of start indices.
    """
    positions = set()
    n = len(generated_ids)
    for candidate_list in pii_values_ids:
        # candidate_list may be a flat list[int] or list[list[int]]
        # normalise to list[list[int]]
        if not candidate_list:
            continue
        if isinstance(candidate_list[0], int):
            sequences = [candidate_list]
        else:
            sequences = candidate_list

        for seq in sequences:
            m = len(seq)
            if m == 0:
                continue
            for i in range(n - m + 1):
                if generated_ids[i:i + m] == seq:
                    positions.add(i)
    return sorted(positions)


def detect_sustained_flags(
    entropy_seq: np.ndarray,
    ceiling: float,
    min_run: int,
) -> list[int]:
    """
    Method 2: Sustained low-entropy window detector.
    Fires at the START of any consecutive run where H_t < ceiling for >= min_run tokens.

    Args:
        entropy_seq: per-token entropy values
        ceiling:     absolute entropy threshold (nats)
        min_run:     minimum consecutive tokens below ceiling to trigger
    Returns:
        list of start indices of qualifying low-entropy runs
    """
    flags = []
    run_start = None
    run_len = 0
    fired_this_run = False

    for t, h in enumerate(entropy_seq):
        if h < ceiling:
            if run_start is None:
                run_start = t
                run_len = 1
                fired_this_run = False
            else:
                run_len += 1
            # Fire exactly once per run, as soon as min_run is reached
            if run_len == min_run and not fired_this_run:
                flags.append(run_start)
                fired_this_run = True
        else:
            run_start = None
            run_len = 0
            fired_this_run = False

    return flags


def compute_lead_time(red_flag_indices: list[int], pii_positions: list[int]) -> list[int]:
    """
    For each PII start position, find the most recent Red Flag before it.
    Lead-time = pii_position - red_flag_index.

    Returns list of lead-time values (positive = flag fired before PII).
    """
    lead_times = []
    for pii_pos in pii_positions:
        preceding = [rf for rf in red_flag_indices if rf <= pii_pos]
        if preceding:
            lead_times.append(pii_pos - max(preceding))
    return lead_times
