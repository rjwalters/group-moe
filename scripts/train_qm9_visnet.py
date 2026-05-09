"""Train ViSNet on QM9 U0 prediction (rigid-equivariance arm for Paper 2).

ViSNet (Wang et al., 2024) is the modern equivariant successor to PaiNN —
vector-scalar interactive equivariant message passing. PyG ships PaiNN-free
but ViSNet-included; for the Paper 2 three-way comparison (SchNet vs
SchNet+GroupMoE vs full equivariance) ViSNet plays the "rigid equivariance"
role.

Headline metric: MAE on internal energy at 0K (U0), reported in meV.
Reference number from the ViSNet paper: ~3.3 meV on QM9 U0 with the full
recipe; literature PaiNN ~5.85 meV is the more conservative reference.

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
from torch_geometric.nn.models import ViSNet

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.qm9 import HARTREE_TO_EV, get_atomref_tensor, load_qm9, make_loaders


TARGET_IDX = 7  # U0


def get_device() -> torch.device:
    # ViSNet's edge graph uses torch_cluster.radius_graph, which has no MPS
    # kernel. Fall back to CPU on Apple silicon for smoke tests; CUDA on Lambda.
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    """Return MAE in eV. ViSNet's forward already de-normalizes (mean/std/atomref handled internally)."""
    model.eval()
    total_abs_err = 0.0
    total_n = 0
    for batch in loader:
        batch = batch.to(device)
        pred, _ = model(batch.z, batch.pos, batch.batch)
        pred = pred.squeeze(-1)
        target = batch.y[:, TARGET_IDX]
        err = (pred - target).abs()
        total_abs_err += err.sum().item()
        total_n += err.numel()
    return total_abs_err / total_n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, default="visnet_baseline")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Peak LR. ViSNet paper uses 1e-4 with warmup; SchNet's 5e-4 caused divergence at epoch 15 in v1.")
    parser.add_argument("--lr-min", type=float, default=1e-7,
                        help="Minimum LR at the end of cosine schedule.")
    parser.add_argument("--warmup-epochs", type=int, default=5,
                        help="Linear LR warmup epochs before cosine decay starts. Required for ViSNet stability.")
    parser.add_argument("--grad-clip", type=float, default=10.0,
                        help="Gradient norm clipping. ViSNet has tensor products + attention — single outlier batch can spike grads. "
                             "Typical grad norm on a fresh ViSNet is ~20; clip at 10 catches outliers without throttling normal updates.")
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--scheduler", choices=["cosine", "plateau"], default="cosine")
    parser.add_argument("--scheduler-patience", type=int, default=80)
    parser.add_argument("--scheduler-factor", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden", type=int, default=256,
                        help="Hidden channels. SchNet v5 baseline used 256; matched here for fair comparison.")
    parser.add_argument("--num-layers", type=int, default=8,
                        help="ViSNet message-passing layers. SchNet v5 used 8 interactions; matched here.")
    parser.add_argument("--num-rbf", type=int, default=20)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--lmax", type=int, default=1,
                        help="Max spherical harmonic degree. lmax=1 = vector features (PaiNN-style).")
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--max-num-neighbors", type=int, default=32)
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
    # ViSNet handles per-atom reference internally via the `atomref` arg, so
    # we pass the residual scale through `mean` and `std` (same as SchNet).
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

    model = ViSNet(
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
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[setup] ViSNet params={n_params:,}", flush=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    if args.scheduler == "cosine":
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
            print(f"[setup] scheduler=linear-warmup({args.warmup_epochs}ep)+cosine, lr {args.lr*1e-3:.1e} -> {args.lr:.1e} -> {args.lr_min:.1e} over {args.epochs} epochs", flush=True)
        else:
            scheduler = cosine
            print(f"[setup] scheduler=cosine, lr {args.lr:.1e} -> {args.lr_min:.1e} over {args.epochs} epochs", flush=True)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.scheduler_factor,
            patience=args.scheduler_patience,
            min_lr=1e-7,
        )
        print(f"[setup] scheduler=plateau, lr={args.lr:.1e}, patience={args.scheduler_patience}, factor={args.scheduler_factor}", flush=True)
    print(f"[setup] grad_clip={args.grad_clip}", flush=True)

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
            pred, _ = model(batch.z, batch.pos, batch.batch)
            pred = pred.squeeze(-1)
            target = batch.y[:, TARGET_IDX]
            loss = F.l1_loss(pred, target)
            # NaN guard: abort fast on divergence (v1 ran 40 NaN epochs before we noticed).
            if not torch.isfinite(loss):
                print(f"[abort] non-finite loss at epoch {epoch}; aborting to save compute.", flush=True)
                sys.exit(2)
            loss.backward()
            if args.grad_clip is not None and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()
            train_loss_sum += loss.item() * pred.numel()
            train_n += pred.numel()
        train_mae = train_loss_sum / train_n

        val_mae = evaluate(model, loaders["val"], device)
        if args.scheduler == "cosine":
            scheduler.step()
        else:
            scheduler.step(val_mae)
        epoch_time = time.time() - t_epoch

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
