"""Train SchNet + GroupMoE on QM9 U0 prediction (selective-equivariance arm).

This is the headline novel contribution of Paper 2: SchNet with one inserted
SO(3)-equivariant Group-MoE block. The router decides per-atom which (if any)
symmetry-type expert to apply, giving the model selective rather than rigid
equivariance.

Recipe matches v4 ViSNet for fair comparison: 1000 epochs cosine + 5-epoch
linear warmup, AdamW 1e-4 → 1e-7, grad-clip 10.0, batch 100, on the
canonical 110k/10k/10.8k QM9 split. Adds load-balancing loss to the main
MAE loss (already weighted by `load_balance_weight` inside the model).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.qm9 import HARTREE_TO_EV, get_atomref_tensor, load_qm9, make_loaders
from src.models.schnet_groupmoe import SchNetGroupMoE


TARGET_IDX = 7  # U0


class CdistRadiusGraph(torch.nn.Module):
    """Device-agnostic radius graph using torch.cdist (MPS fallback)."""

    def __init__(self, cutoff: float):
        super().__init__()
        self.cutoff = cutoff

    def forward(self, pos: torch.Tensor, batch: torch.Tensor):
        dist = torch.cdist(pos, pos)
        same_graph = batch.unsqueeze(0) == batch.unsqueeze(1)
        within = (dist < self.cutoff) & (dist > 0) & same_graph
        edge_index = within.nonzero(as_tuple=False).t().contiguous()
        edge_weight = dist[edge_index[0], edge_index[1]]
        return edge_index, edge_weight


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, dict[str, float]]:
    """Return (val_mae_eV, mean_routing_stats)."""
    model.eval()
    total_abs_err = 0.0
    total_n = 0
    # Routing stats accumulator: sum-then-divide by num batches
    stats_sum: dict[str, float] = {}
    n_batches = 0
    expert_names = [t.name for t in model.symmetry_types]
    for batch in loader:
        batch = batch.to(device)
        pred, decision, _ = model(batch.z, batch.pos, batch.batch)
        pred = pred.squeeze(-1)
        target = batch.y[:, TARGET_IDX]
        err = (pred - target).abs()
        total_abs_err += err.sum().item()
        total_n += err.numel()
        s = model.moe.router.routing_stats(decision, expert_names=expert_names)
        for k, v in s.items():
            stats_sum[k] = stats_sum.get(k, 0.0) + v
        n_batches += 1
    avg_stats = {k: v / max(n_batches, 1) for k, v in stats_sum.items()}
    return total_abs_err / total_n, avg_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, default="groupmoe_baseline")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Peak LR. Matches v4 ViSNet stable recipe.")
    parser.add_argument("--lr-min", type=float, default=1e-7)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden", type=int, default=256,
                        help="SchNet hidden width. Matches v5 baseline / v4 ViSNet for fair comparison.")
    parser.add_argument("--num-interactions", type=int, default=8)
    parser.add_argument("--num-gaussians", type=int, default=20)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--moe-position", type=int, default=None,
                        help="1-indexed position of MoE block. Default: num_interactions // 2.")
    parser.add_argument("--load-balance-weight", type=float, default=0.01,
                        help="Coefficient on the load-balancing loss. Higher = more uniform expert usage.")
    parser.add_argument("--include-irrep-norms", action="store_true",
                        help="If set, the scalar reducer also receives the L2 norms of each l>0 channel "
                             "(rotation-invariant summaries of the equivariant work). Default: scalars only "
                             "(v1 behavior). Set this to test whether the scalar bottleneck is the issue.")
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

    interaction_graph = None if device.type == "cuda" else CdistRadiusGraph(args.cutoff)
    model = SchNetGroupMoE(
        hidden_channels=args.hidden,
        num_filters=args.hidden,
        num_interactions=args.num_interactions,
        num_gaussians=args.num_gaussians,
        cutoff=args.cutoff,
        interaction_graph=interaction_graph,
        atomref=atomref.unsqueeze(-1),
        mean=mean,
        std=std,
        moe_position=args.moe_position,
        load_balance_weight=args.load_balance_weight,
        include_irrep_norms=args.include_irrep_norms,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[setup] SchNetGroupMoE params: {n_params:,}", flush=True)
    print(f"[setup] expert irreps: {model.expert_irreps}", flush=True)
    print(f"[setup] moe_position: {model.moe_position}/{args.num_interactions}", flush=True)
    print(f"[setup] symmetry types: {[t.name for t in model.symmetry_types]}", flush=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    cosine_epochs = max(1, args.epochs - args.warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cosine_epochs, eta_min=args.lr_min,
    )
    if args.warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-3, end_factor=1.0, total_iters=args.warmup_epochs,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[args.warmup_epochs],
        )
        print(f"[setup] scheduler=linear-warmup({args.warmup_epochs}ep)+cosine, "
              f"lr {args.lr*1e-3:.1e} -> {args.lr:.1e} -> {args.lr_min:.1e} over {args.epochs} epochs", flush=True)
    else:
        scheduler = cosine
        print(f"[setup] scheduler=cosine, lr {args.lr:.1e} -> {args.lr_min:.1e} over {args.epochs} epochs", flush=True)
    print(f"[setup] grad_clip={args.grad_clip}, load_balance_weight={args.load_balance_weight}", flush=True)

    log = {
        "config": vars(args),
        "n_params": n_params,
        "device": str(device),
        "mean_eV": mean,
        "std_eV": std,
        "expert_irreps": str(model.expert_irreps),
        "moe_position": model.moe_position,
        "epochs": [],
    }
    best_val_mae = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        t_epoch = time.time()
        train_main_sum = 0.0
        train_lb_sum = 0.0
        train_n = 0
        for batch in loaders["train"]:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred, decision, lb_loss = model(batch.z, batch.pos, batch.batch)
            pred = pred.squeeze(-1)
            target = batch.y[:, TARGET_IDX]
            main_loss = F.l1_loss(pred, target)
            loss = main_loss + lb_loss
            if not torch.isfinite(loss):
                print(f"[abort] non-finite loss at epoch {epoch}; aborting to save compute.", flush=True)
                sys.exit(2)
            loss.backward()
            if args.grad_clip is not None and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()
            train_main_sum += main_loss.item() * pred.numel()
            train_lb_sum += lb_loss.item() * pred.numel()
            train_n += pred.numel()
        train_mae = train_main_sum / train_n
        train_lb = train_lb_sum / train_n

        val_mae, val_routing = evaluate(model, loaders["val"], device)
        scheduler.step()
        epoch_time = time.time() - t_epoch

        if epoch % 10 == 0 or epoch == args.epochs:
            test_mae, test_routing = evaluate(model, loaders["test"], device)
        else:
            test_mae, test_routing = None, None

        rec = {
            "epoch": epoch,
            "train_mae_eV": train_mae,
            "train_lb_loss": train_lb,
            "val_mae_eV": val_mae,
            "val_mae_meV": val_mae * 1000.0,
            "val_routing": val_routing,
            "test_mae_eV": test_mae,
            "test_mae_meV": (test_mae * 1000.0) if test_mae is not None else None,
            "test_routing": test_routing,
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time_sec": epoch_time,
        }
        log["epochs"].append(rec)
        log_path.write_text(json.dumps(log, indent=2))
        # Compact one-line summary for console
        pt_rate = val_routing.get("pass_through_rate", 0)
        print(
            f"[epoch {epoch:3d}] "
            f"train_mae={train_mae*1000:.1f} meV  "
            f"val_mae={val_mae*1000:.1f} meV  "
            f"{'test_mae=' + f'{test_mae*1000:.1f}' + ' meV  ' if test_mae is not None else ''}"
            f"pt={pt_rate:.2f}  "
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
                    "val_routing": val_routing,
                },
                ckpt_path,
            )

    print(f"[done] best val_mae={best_val_mae*1000:.1f} meV (checkpoint: {ckpt_path})", flush=True)


if __name__ == "__main__":
    main()
