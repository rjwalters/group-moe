"""Train ViSNet+GroupMoE on QM9 U0 (Route 2 from paper2_routes_forward).

The v2 sweep showed SchNet+GroupMoE consistently underperforms its host
because the MoE has to reduce equivariant computation back to scalars to
feed SchNet's downstream layers. This script tests whether the MoE works
when the host *itself* is equivariant — ViSNet keeps (scalar, vector)
features end-to-end, and the MoE operates directly on that representation
with no scalar bottleneck.

Recipe matches v4 ViSNet for fair comparison: 1000 epochs cosine + 5-epoch
warmup, AdamW 1e-4 → 1e-7, grad_clip 10, batch 100.

Win condition: matching v4 ViSNet's converged val_mae (~9 meV) at lower
FLOPs (because MoE applies experts only where the router fires, vs
ViSNet's always-on equivariant layers). Even matching at equal FLOPs
would be a positive efficiency-neutral result.
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
from src.models.visnet_groupmoe import VisNetGroupMoE


TARGET_IDX = 7  # U0


def get_device() -> torch.device:
    # ViSNet's edge graph uses torch_cluster.radius_graph (no MPS kernel).
    # CUDA on Lambda; CPU is workable for tiny smoke tests only.
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, dict[str, float]]:
    """Return (val_mae_eV, mean_routing_stats)."""
    model.eval()
    total_abs_err = 0.0
    total_n = 0
    stats_sum: dict[str, float] = {}
    n_batches = 0
    expert_names = [f"expert_{k}" for k in range(model.n_experts)]
    for batch in loader:
        batch = batch.to(device)
        pred, decision, _ = model(batch.z, batch.pos, batch.batch)
        pred = pred.squeeze(-1)
        target = batch.y[:, TARGET_IDX]
        err = (pred - target).abs()
        total_abs_err += err.sum().item()
        total_n += err.numel()
        s = model.representation_model.router.routing_stats(
            decision, expert_names=expert_names,
        )
        for k, v in s.items():
            stats_sum[k] = stats_sum.get(k, 0.0) + v
        n_batches += 1
    avg_stats = {k: v / max(n_batches, 1) for k, v in stats_sum.items()}
    return total_abs_err / total_n, avg_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, default="visnet_groupmoe_v1")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-min", type=float, default=1e-7)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    # ViSNet hyperparameters — matched to v4
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-rbf", type=int, default=20)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--lmax", type=int, default=1)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--max-num-neighbors", type=int, default=32)
    # MoE-specific args
    parser.add_argument("--n-experts", type=int, default=3)
    parser.add_argument("--moe-position", type=int, default=None,
                        help="1-indexed position of MoE block. Default: num_layers // 2.")
    parser.add_argument("--load-balance-weight", type=float, default=0.01)
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

    model = VisNetGroupMoE(
        lmax=args.lmax,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        hidden_channels=args.hidden,
        num_rbf=args.num_rbf,
        cutoff=args.cutoff,
        max_num_neighbors=args.max_num_neighbors,
        atomref=atomref,
        mean=mean,
        std=std,
        derivative=False,
        n_experts=args.n_experts,
        moe_position=args.moe_position,
        load_balance_weight=args.load_balance_weight,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[setup] VisNetGroupMoE params: {n_params:,}", flush=True)
    print(f"[setup] moe_position: {model.moe_position}/{args.num_layers}", flush=True)
    print(f"[setup] n_experts: {args.n_experts}, load_balance_weight: {args.load_balance_weight}", flush=True)

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
    else:
        scheduler = cosine
    print(f"[setup] grad_clip={args.grad_clip}, lr {args.lr:.1e} -> {args.lr_min:.1e} over {args.epochs} epochs", flush=True)

    log = {
        "config": vars(args),
        "n_params": n_params,
        "device": str(device),
        "mean_eV": mean,
        "std_eV": std,
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
                print(f"[abort] non-finite loss at epoch {epoch}", flush=True)
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
