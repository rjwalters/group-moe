#!/bin/bash
# Runs ON the Lambda Labs instance (not locally).
# Sets up the Python environment and runs the requested training command.
#
# Expected layout on the instance after lambda_train.sh has rsync'd:
#   /home/ubuntu/group-moe/                  the repo
#   /lambda/nfs/group-moe-data/              persistent filesystem
#   /lambda/nfs/group-moe-data/data/qm9/     QM9 cache (downloaded once, reused)
#   /lambda/nfs/group-moe-data/runs/<name>/  results.json + best.pt
#
# Args: passed through to scripts/train_qm9.py (--run-name, --epochs, etc.)
#
# Exit code is the training script's exit code.

set -euo pipefail

REPO=/home/ubuntu/group-moe
FS=/lambda/nfs/group-moe-data

# Cache uv's wheel cache on the persistent filesystem so torch-cluster /
# torch-scatter (CUDA wheels, ~10 min to compile) only build once.
export UV_CACHE_DIR="$FS/.uv-cache"

cd "$REPO"

echo "[remote] host: $(hostname)"
echo "[remote] python: $(python3 --version)"
echo "[remote] nvidia driver:"
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader 2>&1 || echo "NO GPU DETECTED"
echo "[remote] cuda runtime (from driver): $(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: [0-9.]+' || echo 'unknown')"
echo "[remote] persistent fs:"
df -h "$FS" | tail -1

# Mirror local layout: data/qm9/ holds both the dataset (raw/, processed/) and
# per-run subdirectories (e.g. schnet_baseline/). All of it lives on the
# persistent filesystem so it survives instance termination.
mkdir -p "$FS/data/qm9" "$UV_CACHE_DIR"
echo "[remote] uv cache dir: $UV_CACHE_DIR ($(du -sh "$UV_CACHE_DIR" 2>/dev/null | awk '{print $1}'))"

# Use uv if available, fall back to pip. Lambda images usually have pip ready.
if ! command -v uv >/dev/null 2>&1; then
    echo "[remote] installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Build venv. We use --system Python (3.10+ on Lambda images) rather than
# pinning to 3.14 so we don't have to compile from source.
if [ ! -d "$REPO/.venv" ]; then
    echo "[remote] creating venv..."
    uv venv --python 3.11
fi

# Lambda A10 instances ship with drivers supporting CUDA 12.8 runtime.
# Install all torch-stack packages in a SINGLE transaction with the cu121 wheel
# index — broad driver compatibility, and prevents a follow-up install (e.g. of
# torch_geometric) from re-resolving torch to the default cu130 wheel.
#
# If FORCE_REBUILD_VENV=1 is set in the environment, we nuke the venv and uv
# cache first. Use this when changing CUDA wheel versions to avoid stale cache
# pollution.
if [ "${FORCE_REBUILD_VENV:-}" = "1" ]; then
    echo "[remote] FORCE_REBUILD_VENV=1: clearing venv + uv cache..."
    rm -rf "$REPO/.venv" "$UV_CACHE_DIR"
    mkdir -p "$UV_CACHE_DIR"
    uv venv --python 3.11
fi

echo "[remote] installing torch stack (cu128) + project deps in one transaction..."
# Lambda A10 ships with NVIDIA driver supporting CUDA 12.8 runtime.
# cu128 is the matching wheel index — torch 2.11+cu128 wheels exist.
#
# Critical: --index-strategy unsafe-best-match. Without it, uv's default
# behavior is "use the first index that has the package" — and PyPI has plain
# torch (cu130 default), so uv stops at PyPI and never checks the cu128 index.
# With unsafe-best-match, uv considers all indexes and picks the version that
# matches the explicit constraint (or the highest available).
#
# Verbose output (no --quiet) so we can see the final wheel hashes.
uv pip install \
    --index-url https://download.pytorch.org/whl/cu128 \
    --extra-index-url https://pypi.org/simple/ \
    --index-strategy unsafe-best-match \
    torch \
    numpy scipy einops matplotlib tqdm \
    torch_geometric e3nn ase 2>&1 | tail -20

echo "[remote] verifying torch CUDA..."
# Use the venv's python directly. Do NOT use `uv run` — that triggers a
# `uv sync` which re-resolves torch from pyproject.toml (no pin) and replaces
# our cu128 install with the default cu130 wheel.
"$REPO/.venv/bin/python" -c "
import torch
print(f'  torch={torch.__version__}')
print(f'  cuda_available={torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  device={torch.cuda.get_device_name(0)}')
    print(f'  cuda_runtime={torch.version.cuda}')
else:
    print('  (CUDA NOT AVAILABLE — training will fall back to CPU)')
"

# torch-cluster + torch-scatter compile against whatever torch is installed.
# Same --index-strategy needed so they can resolve their build deps consistently.
# First build takes ~19 min; subsequent runs hit the persistent uv cache (~1s).
echo "[remote] installing torch-cluster + torch-scatter (CUDA build, cached on persistent fs)..."
uv pip install --no-build-isolation \
    --index-strategy unsafe-best-match \
    torch-cluster torch-scatter

# Symlink the repo's data/qm9 to the persistent filesystem so QM9 only downloads once
# AND so per-run output dirs (data/qm9/<run-name>/) land on the persistent fs.
mkdir -p "$REPO/data"
if [ ! -L "$REPO/data/qm9" ]; then
    rm -rf "$REPO/data/qm9"
    ln -s "$FS/data/qm9" "$REPO/data/qm9"
fi

# Run training. Use the venv's python directly (not `uv run`) to avoid uv-sync
# replacing our pinned-CUDA torch with the default-CUDA wheel.
echo "[remote] launching training: $*"
exec "$REPO/.venv/bin/python" scripts/train_qm9.py "$@"
