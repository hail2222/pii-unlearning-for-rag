#!/bin/bash
# ─────────────────────────────────────────────────────────────
# Server one-time setup script
# Usage:
#   bash setup_server.sh <github_repo_url>
# Example:
#   bash setup_server.sh https://github.com/yourname/entropy-pii-detection
# ─────────────────────────────────────────────────────────────

set -e

REPO_URL=$1
if [ -z "$REPO_URL" ]; then
    echo "Usage: bash setup_server.sh <github_repo_url>"
    exit 1
fi

# ── Step 1: Miniconda ─────────────────────────────────────────
echo "=============================="
echo "Step 1: Install Miniconda"
echo "=============================="

if [ ! -d "$HOME/miniconda3" ]; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p "$HOME/miniconda3"
    rm miniconda.sh
    "$HOME/miniconda3/bin/conda" init bash
    echo "Miniconda installed. Reloading shell..."
else
    echo "Miniconda already installed. Skipping."
fi

export PATH="$HOME/miniconda3/bin:$PATH"

# ── Step 2: Clone repo ────────────────────────────────────────
echo ""
echo "=============================="
echo "Step 2: Clone repository"
echo "=============================="

REPO_NAME=$(basename "$REPO_URL" .git)
if [ ! -d "$HOME/$REPO_NAME" ]; then
    git clone "$REPO_URL" "$HOME/$REPO_NAME"
else
    echo "Repo already exists. Pulling latest..."
    git -C "$HOME/$REPO_NAME" pull
fi

# ── Step 3: Conda env ─────────────────────────────────────────
echo ""
echo "=============================="
echo "Step 3: Create conda env"
echo "=============================="

if conda env list | grep -q "entropy-pii"; then
    echo "Conda env 'entropy-pii' already exists. Skipping."
else
    conda create -n entropy-pii python=3.11 -y
fi

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate entropy-pii

# ── Step 4: Install packages ──────────────────────────────────
echo ""
echo "=============================="
echo "Step 4: Install packages"
echo "=============================="

pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r "$HOME/$REPO_NAME/requirements_server.txt"

# ── Step 5: Verify ────────────────────────────────────────────
echo ""
echo "=============================="
echo "Step 5: Verify GPU & packages"
echo "=============================="

python -c "
import torch
print('PyTorch      :', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU          :', torch.cuda.get_device_name(0))
    print('VRAM (GB)    :', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
import transformers
print('Transformers :', transformers.__version__)
"

echo ""
echo "=============================="
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "  conda activate entropy-pii"
echo "  cd ~/$REPO_NAME"
echo "  export HF_TOKEN=your_token_here"
echo "  bash run_server.sh"
echo "=============================="
