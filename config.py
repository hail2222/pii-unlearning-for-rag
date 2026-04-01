import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../Toward-Practical-PII-Unlearning/data")

PROFILES_PATH    = os.path.join(DATA_DIR, "PII/full_user_profiles.json")
WORLD_FACTS_PATH = os.path.join(DATA_DIR, "test/world_facts_perturbed.json")
RESULTS_DIR      = os.path.join(BASE_DIR, "results")

# ── Model ──────────────────────────────────────────────────────────────────────
# Local debug: "Qwen/Qwen2.5-0.5B-Instruct"
# Remote experiment: "meta-llama/Llama-3.1-8B-Instruct" or "Qwen/Qwen2.5-7B-Instruct"
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# Hidden state layer to extract (index from 0)
# Qwen2.5-0.5B has 24 layers → use layer 18 (upper-middle)
# Llama-3.1-8B has 32 layers → use layer 24
PROBE_LAYER = 18

# ── Generation ─────────────────────────────────────────────────────────────────
MAX_NEW_TOKENS = 80
DEVICE = "cpu"   # "cuda" on remote server

# ── Entropy detection ──────────────────────────────────────────────────────────
# Method 1 (original): Adaptive ΔH drop
# Flag when ΔH_t < mean(ΔH) - ENTROPY_DROP_ZSCORE * std(ΔH)
ENTROPY_DROP_THRESHOLD = 1.0   # fallback fixed threshold (nats)
ENTROPY_DROP_ZSCORE    = 2.0   # adaptive: flag drops > 2 std below mean

# Method 2 (new): Sustained low-entropy window
# Flag when H_t stays below ABS_ENTROPY_CEILING for at least MIN_RUN_LENGTH consecutive tokens
ABS_ENTROPY_CEILING = 0.5    # nats — "low entropy" absolute cutoff
MIN_RUN_LENGTH      = 10     # minimum consecutive tokens below ceiling to fire

# ── Experiment ─────────────────────────────────────────────────────────────────
# Number of samples per condition (reduce for quick local debug)
MAX_SAMPLES = 20   # set to None to use all
