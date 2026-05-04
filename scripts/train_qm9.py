"""Train SchNet on QM9 U0 prediction.

Headline metric: MAE on internal energy at 0K (U0), reported in meV.
Reference number from the SchNet paper: ~14 meV with full training.

Logs per-epoch train/val/test loss to data/qm9/<run_name>/results.json
and saves the best checkpoint by validation MAE.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.nn import SchNet

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.qm9 import HARTREE_TO_EV, get_atomref_tensor, load_qm9, make_loaders


TARGET_IDX = 7  # U0


class CdistRadiusGraph(torch.nn.Module):
    """Device-agnostic radius graph using torch.cdist.

    PyG's default uses torch_cluster.radius_graph which is CPU-only and
    incompatible with MPS. For QM9 (max ~29 atoms per molecule) the O(N^2)
    cost is negligible.
    """

    def __init__(self, cutoff: float):
        super().__init__()
        self.cutoff = cutoff

    def forward(self, pos: torch.Tensor, batch: torch.Tensor):
        # pairwise distances within the same graph only
        dist = torch.cdist(pos, pos)
        same_graph = batch.unsqueeze(0) == batch.unsqueeze(1)
        within_cutoff = (dist < self.cutoff) & (dist > 0) & same_graph
        edge_index = within_cutoff.nonzero(as_tuple=False).t().contiguous()
        edge_weight = dist[edge_index[0], edge_index[1]]
        return edge_index, edge_weight


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    """Return MAE in eV. SchNet's forward already de-normalizes (mean/std/atomref handled internally)."""
    model.eval()
    total_abs_err = 0.0
    total_n = 0
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch.z, batch.pos, batch.batch).squeeze(-1)
        target = batch.y[:, TARGET_IDX]
        err = (pred - target).abs()
        total_abs_err += err.sum().item()
        total_n += err.numel()
    return total_abs_err / total_n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, default="schnet_baseline")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--scheduler-patience", type=int, default=80)
    parser.add_argument("--scheduler-factor", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--num-interactions", type=int, default=6)
    parser.add_argument("--num-gaussians", type=int, default=20)
    parser.add_argument("--cutoff", type=float, default=5.0)
    args = parser.parse_args()

    out_dir = Path("data/qm9") / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "results.json"
    ckpt_path = out_dir / "best.pt"

    device = get_device()
    print(f"[setup] device={device}", flush=True)
    torch.manual_seed(args.seed)

    print("[setup] loading QM9 ...", flush=True)
    t0 = time.time()
    dataset, split = load_qm9(seed=args.seed)
    loaders = make_loaders(dataset, split, batch_size=args.batch_size)
    print(f"[setup] dataset loaded in {time.time()-t0:.1f}s, "
          f"train={len(split['train'])}, val={len(split['val'])}, test={len(split['test'])}",
          flush=True)

    # Compute mean and std of the (atomref-corrected) U0 over the train set.
    # PyG's SchNet handles per-atom reference internally via the `atomref` arg,
    # so we pass the residual scale through `mean` and `std`.
    atomref = get_atomref_tensor()
    train_targets = []
    for i in split["train"]:
        d = dataset[i.item()]
        residual = d.y[0, TARGET_IDX] - atomref[d.z].sum()
        train_targets.append(residual.item())
    train_targets = torch.tensor(train_targets)
    mean = train_targets.mean().item()
    std = train_targets.std().item()
    print(f"[setup] residual U0 mean={mean:.3f} eV, std={std:.3f} eV", flush=True)

    # On CUDA, use the much faster torch_cluster.radius_graph (default).
    # On MPS, torch_cluster is CPU-only and incompatible — use a cdist fallback.
    interaction_graph = None if device.type == "cuda" else CdistRadiusGraph(args.cutoff)

    model = SchNet(
        hidden_channels=args.hidden,
        num_filters=args.hidden,
        num_interactions=args.num_interactions,
        num_gaussians=args.num_gaussians,
        cutoff=args.cutoff,
        interaction_graph=interaction_graph,
        atomref=atomref.unsqueeze(-1),
        mean=mean,
        std=std,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[setup] SchNet params={n_params:,}", flush=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        min_lr=1e-7,
    )

    log = {
        "config": vars(args),
        "n_params": n_params,
        "device": str(device),
        "mean_eV": mean,
        "std_eV": std,
        "epochs": [],
    }
    best_val_mae = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        t_epoch = time.time()
        train_loss_sum = 0.0
        train_n = 0
        for batch in loaders["train"]:
            batch = batch.to(device)
            optimizer.zero_grad()
            # SchNet returns predictions already de-normalized using mean/std
            # and offset by atomref. So compare directly to batch.y[:, TARGET_IDX].
            pred = model(batch.z, batch.pos, batch.batch).squeeze(-1)
            target = batch.y[:, TARGET_IDX]
            loss = F.l1_loss(pred, target)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * pred.numel()
            train_n += pred.numel()
        train_mae = train_loss_sum / train_n

        val_mae = evaluate(model, loaders["val"], device)
        scheduler.step(val_mae)
        epoch_time = time.time() - t_epoch

        # Test only every 10 epochs to save compute (long run).
        if epoch % 10 == 0 or epoch == args.epochs:
            test_mae = evaluate(model, loaders["test"], device)
        else:
            test_mae = None

        rec = {
            "epoch": epoch,
            "train_mae_eV": train_mae,
            "val_mae_eV": val_mae,
            "val_mae_meV": val_mae * 1000.0,
            "test_mae_eV": test_mae,
            "test_mae_meV": (test_mae * 1000.0) if test_mae is not None else None,
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time_sec": epoch_time,
        }
        log["epochs"].append(rec)
        log_path.write_text(json.dumps(log, indent=2))
        print(
            f"[epoch {epoch:3d}] "
            f"train_mae={train_mae*1000:.1f} meV  "
            f"val_mae={val_mae*1000:.1f} meV  "
            f"{'test_mae=' + f'{test_mae*1000:.1f}' + ' meV  ' if test_mae is not None else ''}"
            f"lr={optimizer.param_groups[0]['lr']:.2e}  "
            f"({epoch_time:.1f}s)",
            flush=True,
        )

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": vars(args),
                    "epoch": epoch,
                    "val_mae_eV": val_mae,
                },
                ckpt_path,
            )

    print(f"[done] best val_mae={best_val_mae*1000:.1f} meV (checkpoint: {ckpt_path})", flush=True)


if __name__ == "__main__":
    main()
