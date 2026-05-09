"""Train SchNet+GroupMoE on MD17 (energy + force prediction).

Why MD17: forces are vectors. Predicting them tests selective equivariance on
a task where equivariance unambiguously matters (unlike QM9 energy, which is
already invariant). See `docs/paper2_routes_forward.md` Route 1.

Forces come for free via autograd of energy w.r.t. positions:
    pos.requires_grad_(True)
    E = model(z, pos, batch)
    F_pred = -torch.autograd.grad(E.sum(), pos, create_graph=True)[0]

This `create_graph=True` lets us backprop through forces during training. It
substantially increases memory use vs energy-only training.

Joint loss (Schütt et al. SchNet paper convention): α * energy_loss + β * force_loss
with α=0.05, β=0.95 — forces dominate. Headline metric is force MAE in
kcal/mol/Å (literature convention for MD17), reported alongside energy MAE.

Defaults are tuned for **local Mac M3 Ultra** runs:
  - One molecule at a time (per-run-name)
  - Smaller model (h=128, num_interactions=4) than QM9
  - Reduced eval cadence to keep epoch time tolerable
  - Test eval on a subsampled subset; full test only at end
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

from src.data.md17 import (
    KCAL_PER_MOL_TO_EV,
    load_md17,
    make_loaders,
    make_split,
    normalize_force_field_name,
)
from src.models.schnet_groupmoe import SchNetGroupMoE


# MD17 native units: energy kcal/mol, force kcal/mol/Å. We report both
# directly (matches literature) and convert energy MAE to meV for cross-task
# comparison with our QM9 numbers.


class CdistRadiusGraph(torch.nn.Module):
    """Device-agnostic radius graph for MPS.

    Uses pairwise subtract-and-norm rather than `torch.cdist` because
    `aten::_cdist_backward` has no MPS kernel — and we need gradients
    through this layer for force prediction (forces = -dE/dpos requires
    backprop through edge construction).

    For QM9-scale graphs (≤30 atoms per molecule, batched ~100 mols)
    the O(N^2) cost is negligible vs the message-passing layers.
    """

    def __init__(self, cutoff: float):
        super().__init__()
        self.cutoff = cutoff
        self._cutoff_sq = cutoff * cutoff

    def forward(self, pos: torch.Tensor, batch: torch.Tensor):
        # Squared distances from manual subtract; avoids cdist's missing MPS backward.
        diff = pos.unsqueeze(0) - pos.unsqueeze(1)         # (N, N, 3)
        dist_sq = (diff * diff).sum(dim=-1)                # (N, N)
        same_graph = batch.unsqueeze(0) == batch.unsqueeze(1)
        n = pos.shape[0]
        not_self = ~torch.eye(n, dtype=torch.bool, device=pos.device)
        within = (dist_sq < self._cutoff_sq) & not_self & same_graph
        edge_index = within.nonzero(as_tuple=False).t().contiguous()
        # sqrt only on the surviving entries: avoids sqrt(0) gradient explosion at the diagonal.
        edge_weight = dist_sq[edge_index[0], edge_index[1]].sqrt()
        return edge_index, edge_weight


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_forces(model, batch, device, *, create_graph: bool):
    """Forward pass with force prediction via -dE/dpos.

    Returns (energy_pred, force_pred, decision, lb_loss).
    `create_graph=True` during training (so force loss is differentiable);
    `False` during evaluation (saves memory).
    """
    pos = batch.pos.to(device).detach().requires_grad_(True)
    z = batch.z.to(device)
    batch_idx = batch.batch.to(device)

    energy, decision, lb_loss = model(z, pos, batch_idx)
    grad_outputs = torch.ones_like(energy)
    forces = -torch.autograd.grad(
        outputs=[energy],
        inputs=[pos],
        grad_outputs=[grad_outputs],
        create_graph=create_graph,
        retain_graph=create_graph,
    )[0]
    return energy, forces, decision, lb_loss


def evaluate(model, loader, device, force_field: str, n_max: int | None = None):
    """Return (energy_mae_kcal/mol, force_mae_kcal/mol/A) and routing stats.

    `n_max` caps the number of batches evaluated (for fast monitoring on the
    huge MD17 test set). None = evaluate all batches.
    """
    model.eval()
    e_err = 0.0
    e_n = 0
    f_err = 0.0
    f_n = 0
    stats_sum: dict[str, float] = {}
    n_batches = 0
    expert_names = [t.name for t in model.symmetry_types]

    for i, batch in enumerate(loader):
        if n_max is not None and i >= n_max:
            break
        # Force prediction needs grads on pos even at eval time.
        e_pred, f_pred, decision, _ = compute_forces(model, batch, device, create_graph=False)
        e_pred = e_pred.squeeze(-1).detach()
        f_pred = f_pred.detach()
        e_true = batch.energy.to(device)
        f_true = getattr(batch, force_field).to(device)
        e_err += (e_pred - e_true).abs().sum().item()
        e_n += e_pred.numel()
        f_err += (f_pred - f_true).abs().sum().item()
        f_n += f_pred.numel()
        s = model.moe.router.routing_stats(decision, expert_names=expert_names)
        for k, v in s.items():
            stats_sum[k] = stats_sum.get(k, 0.0) + v
        n_batches += 1

    avg_stats = {k: v / max(n_batches, 1) for k, v in stats_sum.items()}
    return e_err / e_n, f_err / f_n, avg_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--molecule", type=str, default="ethanol",
                        help="MD17 molecule name. e.g. ethanol, aspirin, naphthalene.")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Output dir name. Default: groupmoe_md17_<molecule>.")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-min", type=float, default=1e-6)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-train", type=int, default=950,
                        help="Training conformations. SchNet paper uses 1000; we leave 50 for val to fit in 1000-test convention if needed.")
    parser.add_argument("--n-val", type=int, default=50)
    # Model — defaults sized for Mac M3 Ultra
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--num-interactions", type=int, default=4)
    parser.add_argument("--num-gaussians", type=int, default=20)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--moe-position", type=int, default=None,
                        help="Default: num_interactions // 2.")
    parser.add_argument("--load-balance-weight", type=float, default=0.01)
    parser.add_argument("--include-irrep-norms", action="store_true")
    # Joint loss weights — Schütt SchNet paper standard
    parser.add_argument("--energy-weight", type=float, default=0.05)
    parser.add_argument("--force-weight", type=float, default=0.95)
    # Test eval — subsample for monitoring; full eval only at end
    parser.add_argument("--test-batches-per-eval", type=int, default=200,
                        help="Cap test eval to this many batches mid-training (every 10 epochs).")
    parser.add_argument("--final-test-batches", type=int, default=2000,
                        help="Cap test batches for the final eval at run end. MD17 test sets have "
                             "hundreds of thousands of conformations; uncapped final eval would take "
                             "many hours on Mac. 2000 batches × 4 = 8000 conformations is plenty for "
                             "a stable estimate.")
    args = parser.parse_args()

    run_name = args.run_name or f"groupmoe_md17_{args.molecule.replace(' ', '_')}"
    out_dir = Path("data/md17") / "results" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "results.json"
    ckpt_path = out_dir / "best.pt"

    device = get_device()
    print(f"[setup] device={device}", flush=True)
    torch.manual_seed(args.seed)

    print(f"[setup] loading MD17 ({args.molecule}) ...", flush=True)
    t0 = time.time()
    dataset = load_md17(args.molecule)
    n = len(dataset)
    split = make_split(n, n_train=args.n_train, n_val=args.n_val, seed=args.seed)
    loaders = make_loaders(dataset, split, batch_size=args.batch_size)
    force_field = normalize_force_field_name(dataset[0])
    print(f"[setup] dataset loaded in {time.time()-t0:.1f}s, "
          f"train={len(split['train'])}, val={len(split['val'])}, test={len(split['test'])}, "
          f"force_field='{force_field}'",
          flush=True)

    # Energy normalization. Subtle point: PyG's SchNet adds `mean` to its
    # per-atom output and then sum-pools across atoms, so passing the
    # per-molecule mean would offset by n_atoms × mean at the readout. For
    # MD17 (constant atom count per run), divide mean by n_atoms so the
    # readout's sum recovers the per-molecule mean cleanly. (For QM9 with
    # variable atom counts, atomref handles per-atom offsets and `mean` is
    # the residual; that's a different regime.)
    train_energies = torch.tensor([dataset[i.item()].energy.item() for i in split["train"]])
    mean_per_molecule = train_energies.mean().item()
    std_per_molecule = train_energies.std().item()
    n_atoms_per_mol = dataset[0].z.numel()
    mean = mean_per_molecule / n_atoms_per_mol  # per-atom: SchNet sums across atoms
    std = std_per_molecule  # leave unscaled; model learns per-atom scale during training
    print(
        f"[setup] energy mean={mean_per_molecule:.3f} kcal/mol/molecule "
        f"({mean:.3f}/atom × {n_atoms_per_mol} atoms), std={std:.4f} kcal/mol",
        flush=True,
    )

    interaction_graph = None if device.type == "cuda" else CdistRadiusGraph(args.cutoff)
    model = SchNetGroupMoE(
        hidden_channels=args.hidden,
        num_filters=args.hidden,
        num_interactions=args.num_interactions,
        num_gaussians=args.num_gaussians,
        cutoff=args.cutoff,
        interaction_graph=interaction_graph,
        atomref=None,  # MD17: same molecule, no per-element offset needed
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
    print(f"[setup] joint loss: {args.energy_weight} * E_loss + {args.force_weight} * F_loss", flush=True)

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
        "energy_mean_kcalmol": mean,
        "energy_std_kcalmol": std,
        "force_field": force_field,
        "epochs": [],
    }
    best_val_force_mae = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        t_epoch = time.time()
        train_e_err = 0.0
        train_f_err = 0.0
        train_e_n = 0
        train_f_n = 0
        for batch in loaders["train"]:
            optimizer.zero_grad()
            e_pred, f_pred, decision, lb_loss = compute_forces(
                model, batch, device, create_graph=True
            )
            e_pred = e_pred.squeeze(-1)
            e_true = batch.energy.to(device)
            f_true = getattr(batch, force_field).to(device)
            e_loss = F.l1_loss(e_pred, e_true)
            f_loss = F.l1_loss(f_pred, f_true)
            loss = args.energy_weight * e_loss + args.force_weight * f_loss + lb_loss
            if not torch.isfinite(loss):
                print(f"[abort] non-finite loss at epoch {epoch}", flush=True)
                sys.exit(2)
            loss.backward()
            if args.grad_clip is not None and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()
            train_e_err += e_loss.item() * e_pred.numel()
            train_f_err += f_loss.item() * f_pred.numel()
            train_e_n += e_pred.numel()
            train_f_n += f_pred.numel()
        train_e_mae = train_e_err / train_e_n
        train_f_mae = train_f_err / train_f_n

        val_e_mae, val_f_mae, val_routing = evaluate(
            model, loaders["val"], device, force_field, n_max=None
        )
        scheduler.step()
        epoch_time = time.time() - t_epoch

        if epoch % 10 == 0 or epoch == args.epochs:
            test_e_mae, test_f_mae, _ = evaluate(
                model, loaders["test"], device, force_field,
                n_max=args.test_batches_per_eval,
            )
        else:
            test_e_mae, test_f_mae = None, None

        # MPS leaks autograd graph state across iterations when force training
        # uses `create_graph=True`. Wall time crept 240s → 820s over 15 epochs
        # without this. Empty cache + collect at end of each epoch keeps it flat.
        if device.type == "mps":
            torch.mps.empty_cache()

        rec = {
            "epoch": epoch,
            "train_e_mae_kcalmol": train_e_mae,
            "train_f_mae_kcalmol_A": train_f_mae,
            "val_e_mae_kcalmol": val_e_mae,
            "val_f_mae_kcalmol_A": val_f_mae,
            "test_e_mae_kcalmol": test_e_mae,
            "test_f_mae_kcalmol_A": test_f_mae,
            "val_routing": val_routing,
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time_sec": epoch_time,
        }
        log["epochs"].append(rec)
        log_path.write_text(json.dumps(log, indent=2))
        pt_rate = val_routing.get("pass_through_rate", 0)
        print(
            f"[epoch {epoch:3d}] "
            f"train_F={train_f_mae:.3f}  val_F={val_f_mae:.3f}  "
            f"{'test_F=' + f'{test_f_mae:.3f}' + '  ' if test_f_mae is not None else ''}"
            f"train_E={train_e_mae:.3f}  val_E={val_e_mae:.3f}  "
            f"pt={pt_rate:.2f}  lr={optimizer.param_groups[0]['lr']:.2e}  "
            f"({epoch_time:.1f}s)",
            flush=True,
        )

        if val_f_mae < best_val_force_mae:
            best_val_force_mae = val_f_mae
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": vars(args),
                    "epoch": epoch,
                    "val_f_mae_kcalmol_A": val_f_mae,
                    "val_e_mae_kcalmol": val_e_mae,
                    "val_routing": val_routing,
                },
                ckpt_path,
            )

    # Final test eval (capped — see --final-test-batches help text)
    print(f"[final] test eval ({args.final_test_batches} batches × {args.batch_size} = "
          f"{args.final_test_batches * args.batch_size} conformations)...", flush=True)
    test_e_mae, test_f_mae, test_routing = evaluate(
        model, loaders["test"], device, force_field, n_max=args.final_test_batches
    )
    log["final_test"] = {
        "energy_mae_kcalmol": test_e_mae,
        "force_mae_kcalmol_A": test_f_mae,
        "routing": test_routing,
    }
    log_path.write_text(json.dumps(log, indent=2))
    print(
        f"[done] final test: F_MAE={test_f_mae:.3f} kcal/mol/A, E_MAE={test_e_mae:.3f} kcal/mol  "
        f"(best val_F={best_val_force_mae:.3f} kcal/mol/A)",
        flush=True,
    )


if __name__ == "__main__":
    main()
