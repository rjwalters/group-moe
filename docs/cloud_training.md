# Cloud training on Lambda Labs

We train Paper 2 baselines on Lambda Labs A100 / A10 GPUs. The local M1 (MPS backend) hits the wall on long SchNet runs — see `data/qm9/schnet_baseline_v2_failed/` for what went wrong (LR oscillation, MPS instability under contention).

## One-time setup

Already done — captured here so the project is reproducible:

1. **Lambda account + API key.** Provisioned via the sibling `ml-audio-codecs` project. Key lives in `.env` (gitignored).
2. **SSH key** named `mac-studio` registered with Lambda. Used by `lambda.sh launch` automatically.
3. **Persistent filesystem** `group-moe-data` in `us-west-1` (id `aeecfd3315134662bef81274579749b7`). Holds the QM9 dataset and per-run outputs across instance terminations. Storage cost: $0.20/GB/month — trivial for QM9.

If any of these are missing, recreate with:

```bash
# create filesystem (only once; check `./scripts/lambda.sh filesystems` first)
source .env
curl -s -u "$LAMBDA_API_KEY:" -X POST \
    https://cloud.lambda.ai/api/v1/filesystems \
    -H "Content-Type: application/json" \
    -d '{"name": "group-moe-data", "region": "us-west-1"}' | jq .
```

## Two-script architecture

| Script | Where it runs | Purpose |
|---|---|---|
| `scripts/lambda.sh` | local | low-level Lambda CLI: list / launch / ssh / wait-ready / terminate |
| `scripts/lambda_train.sh` | local | high-level workflow: launch → rsync repo → run training → rsync results → terminate |
| `scripts/lambda_remote_setup.sh` | on the instance | install deps, symlink data dir to persistent fs, exec the training command |

You normally only call `lambda_train.sh`. `lambda.sh` is for poking around (capacity check, manual SSH, terminating an orphan).

## Daily workflow

```bash
# Capacity check before a run — Lambda A100s are scarce
./scripts/lambda.sh types

# Smoke test (cheap, short): A10 in us-west-1 for 5 epochs
./scripts/lambda_train.sh gpu_1x_a10 -- --run-name schnet_smoke --epochs 5

# Real baseline: A100 if available, else A10. Defaults to 500 epochs.
./scripts/lambda_train.sh gpu_1x_a100_sxm4 -- --run-name schnet_baseline --epochs 500
```

The script will:

1. Print the cost preview (instance type, $/hr, run name) and ask for confirmation.
2. Launch the instance with the `group-moe-data` filesystem mounted.
3. Block until SSH responds.
4. `rsync` the repo (excluding `.venv`, `data/`, `paper/`, `.git`, etc.).
5. `ssh` in and run `lambda_remote_setup.sh` — installs deps, exec's `train_qm9.py`. Output streams back to your local terminal.
6. `rsync` `data/qm9/<run-name>/` back to your local machine.
7. **Always terminate the instance** — even on Ctrl-C or script crash, via a trap on EXIT.

## Safety

The `EXIT` trap in `lambda_train.sh` reads the launched instance ID from a state file (`.lambda-state/current-instance`) and terminates it. This means:

- Ctrl-C still terminates the instance.
- Script crash still terminates the instance.
- Network drop while SSH'd in still terminates the instance (script returns, trap fires).

If you ever suspect an orphaned instance:

```bash
./scripts/lambda.sh instances    # see what's running
./scripts/lambda.sh terminate <id>
```

To **disable auto-terminate** (for debugging — leave the instance up to inspect):

```bash
LAMBDA_NO_TERMINATE=1 ./scripts/lambda_train.sh gpu_1x_a10 -- --run-name debug --epochs 1
# When done: ./scripts/lambda.sh terminate <id>
```

## Persistent filesystem layout

After the first run, the filesystem looks like:

```
/lambda/nfs/group-moe-data/
├── data/qm9/
│   ├── raw/                    # downloaded by torch_geometric (one-time, ~500MB)
│   ├── processed/              # cached PyG tensors (one-time, ~200MB)
│   ├── schnet_baseline/        # results.json, best.pt
│   ├── schnet_groupmoe/        # future
│   └── ...
```

QM9 only downloads once (~3 minutes the first time). All subsequent runs read from the cache.

## Cost estimates

| Run | Instance | Per-epoch (est.) | 500 epochs | Total |
|---|---|---|---|---|
| Local M1 MPS (failed) | n/a | 350-500s | n/a | failed |
| A10 | gpu_1x_a10 ($1.29/hr) | ~60s | ~8 hr | ~$10 |
| A100 SXM4 | gpu_1x_a100_sxm4 ($1.99/hr) | ~30s | ~4 hr | ~$8 |

A 5-epoch smoke test is **under $0.20**.

## Region constraint

Filesystems are region-locked. Our `group-moe-data` is in `us-west-1`. If A100 capacity is only available in another region (`./scripts/lambda.sh types` shows current state), you have two options:

1. Use the A10 in `us-west-1` instead — fine for SchNet at our scale.
2. Create a duplicate filesystem in the region that has capacity — incurs duplicate storage cost (still cents/month).

## Known gotchas

- `torch_cluster.radius_graph` is CPU-only and crashes on MPS — `train_qm9.py` falls back to a `cdist`-based radius graph when `device.type != "cuda"`. On the Lambda CUDA box it uses the fast kernel automatically.
- The first remote run takes ~5 minutes longer than subsequent runs because `torch-cluster` and `torch-scatter` compile from source (`--no-build-isolation`).
- Lambda's stock Python is 3.10/3.11. We pin to 3.11 in `lambda_remote_setup.sh` to avoid building 3.14 from source.
