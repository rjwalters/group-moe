"""Scaling experiment: accuracy vs num_range.

Sweeps num_range to find the crossover where memorization (LUT) fails
but Group-MoE's algebraic structure continues to generalize.

As num_range grows:
- Number of distinct triples: C(n,3) = n(n-1)(n-2)/6
- Training examples (complement): C(n,3) + n(n-1) + n (one per multiset)
- Test examples: 5*C(n,3) + 2*n(n-1) (remaining orderings)
- LUT size the model must learn: grows as O(n^3)
- Group expert's irrep space: fixed at k=4 (independent of n)

The prediction: at small n, both models memorize the LUT. At large n,
only GroupMoE maintains complement transfer because it compresses
the table through the irrep basis.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.ternary import TernaryDataset, OP_SYM, OP_NONSYM
from src.models.ternary import TernaryGroupMoE, TernaryStandardMoE, TernaryBaseline
from itertools import permutations as perms


def collate(batch):
    return {k: torch.stack([item[k] for item in batch]) for k in batch[0]}


def balance_loss(decision, n_options):
    logits = decision.logits
    probs = F.softmax(logits, dim=-1)
    best = logits.argmax(dim=-1)
    one_hot = F.one_hot(best, n_options).float()
    return n_options * (one_hot.mean(0) * probs.mean(0)).sum()


def evaluate_complement(model, loader, denormalize_fn, device, train_triples):
    model.eval()
    all_preds, all_targets, all_ops = [], [], []
    all_a, all_b, all_c = [], [], []

    with torch.no_grad():
        for batch in loader:
            pred, _ = model(
                batch["a"].to(device), batch["op"].to(device),
                batch["b"].to(device), batch["c"].to(device),
            )
            all_preds.append(denormalize_fn(pred.cpu()))
            all_targets.append(batch["target_raw"])
            all_ops.append(batch["op"])
            all_a.append(batch["a"])
            all_b.append(batch["b"])
            all_c.append(batch["c"])

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    ops = torch.cat(all_ops)
    a_all, b_all, c_all = torch.cat(all_a), torch.cat(all_b), torch.cat(all_c)

    errors = torch.abs(preds - targets)
    sym_mask = ops == OP_SYM

    results = {
        "accuracy": (errors < 0.5).float().mean().item(),
        "mae": errors.mean().item(),
    }

    if sym_mask.any():
        results["sym_accuracy"] = (errors[sym_mask] < 0.5).float().mean().item()
        results["sym_mae"] = errors[sym_mask].mean().item()

        # Complement accuracy
        has_comp = torch.tensor([
            any((p[0], p[1], p[2]) in train_triples
                for p in perms([a_all[i].item(), b_all[i].item(), c_all[i].item()]))
            for i in range(len(a_all))
        ])
        comp_mask = has_comp & sym_mask
        if comp_mask.any():
            results["complement_acc"] = (errors[comp_mask] < 0.5).float().mean().item()
            results["complement_mae"] = errors[comp_mask].mean().item()
            results["n_complement"] = int(comp_mask.sum().item())

    return results


def train_and_evaluate(model_type, num_range, args, device):
    """Train a model and return final complement accuracy."""
    ds_kwargs = dict(
        num_range=num_range, seed=args.seed,
        train_frac=args.train_frac, split_mode="complement",
    )
    train_ds = TernaryDataset(split="train", **ds_kwargs)
    stats = train_ds.get_stats()
    val_ds = TernaryDataset(split="val", **ds_kwargs, stats=stats)
    test_ds = TernaryDataset(split="test", **ds_kwargs, stats=stats)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, collate_fn=collate)

    train_triples = train_ds.symmetric_triples()

    model_kwargs = dict(d_model=args.d_model, n_numbers=num_range, n_blocks=args.n_blocks)
    if model_type == "groupmoe":
        model = TernaryGroupMoE(**model_kwargs).to(device)
    elif model_type == "standardmoe":
        model = TernaryStandardMoE(**model_kwargs).to(device)
    else:
        model = TernaryBaseline(**model_kwargs).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    use_balance = model_type in ("groupmoe", "standardmoe") and args.balance_alpha > 0
    if use_balance:
        moe_layer = model.group_moe if hasattr(model, 'group_moe') else model.standard_moe
        n_opts = moe_layer.n_options if hasattr(moe_layer, 'n_options') else moe_layer.router.n_options

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            a = batch["a"].to(device)
            op = batch["op"].to(device)
            b = batch["b"].to(device)
            c = batch["c"].to(device)
            target = batch["target"].to(device)

            pred, decision = model(a, op, b, c)
            loss = F.mse_loss(pred, target)
            if use_balance and decision is not None:
                loss = loss + args.balance_alpha * balance_loss(decision, n_opts)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        val_results = evaluate_complement(model, val_loader, train_ds.denormalize, device, train_triples)
        if val_results.get("mae", float("inf")) < best_val_loss:
            best_val_loss = val_results["mae"]
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                break

    test_results = evaluate_complement(model, test_loader, train_ds.denormalize, device, train_triples)
    return test_results


def main():
    parser = argparse.ArgumentParser(description="Scaling sweep: accuracy vs num_range")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-blocks", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=75)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.5)
    parser.add_argument("--balance-alpha", type=float, default=0.01)
    parser.add_argument("--output-dir", type=str, default="data/scaling_sweep")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-ranges", type=str, default="10,15,20,25,30,40",
                       help="Comma-separated list of num_range values")
    parser.add_argument("--models", type=str, default="groupmoe,standardmoe,baseline",
                       help="Comma-separated list of model types")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("mps") if torch.backends.mps.is_available() else
        torch.device("cpu"))
    print(f"Device: {device}")

    num_ranges = [int(x) for x in args.num_ranges.split(",")]
    model_types = args.models.split(",")

    results = {mt: {} for mt in model_types}

    for nr in num_ranges:
        n_triples = nr * (nr - 1) * (nr - 2) // 6
        n_train_sym = n_triples + nr * (nr - 1) + nr
        n_test_sym = 5 * n_triples + 2 * nr * (nr - 1)
        print(f"\n{'='*60}")
        print(f"num_range={nr}: {n_triples} distinct triples, "
              f"{n_train_sym} train sym, {n_test_sym} test sym")
        print(f"{'='*60}")

        for mt in model_types:
            torch.manual_seed(args.seed)
            print(f"  Training {mt}...", end=" ", flush=True)
            test_results = train_and_evaluate(mt, nr, args, device)
            comp_acc = test_results.get("complement_acc", 0)
            comp_mae = test_results.get("complement_mae", 0)
            print(f"comp_acc={comp_acc:.4f} mae={comp_mae:.2f}")
            results[mt][nr] = test_results

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print summary table
    print(f"\n{'='*60}")
    print("SCALING SUMMARY: Complement Transfer Accuracy")
    print(f"{'='*60}")
    header = f"{'num_range':>10s}"
    for mt in model_types:
        header += f"  {mt:>14s}"
    header += f"  {'n_triples':>10s}"
    print(header)
    for nr in num_ranges:
        n_triples = nr * (nr - 1) * (nr - 2) // 6
        line = f"{nr:>10d}"
        for mt in model_types:
            acc = results[mt].get(nr, {}).get("complement_acc", 0)
            line += f"  {acc:>14.4f}"
        line += f"  {n_triples:>10d}"
        print(line)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {"groupmoe": "blue", "standardmoe": "orange", "baseline": "green"}

    ax = axes[0]
    for mt in model_types:
        nrs = sorted(results[mt].keys())
        accs = [results[mt][nr].get("complement_acc", 0) for nr in nrs]
        ax.plot(nrs, accs, "o-", label=mt, color=colors.get(mt, None))
    ax.set_xlabel("num_range (n)")
    ax.set_ylabel("Complement Transfer Accuracy")
    ax.set_title("Scaling: Accuracy vs Problem Size")
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.3)

    ax = axes[1]
    for mt in model_types:
        nrs = sorted(results[mt].keys())
        maes = [results[mt][nr].get("complement_mae", 0) for nr in nrs]
        ax.plot(nrs, maes, "o-", label=mt, color=colors.get(mt, None))
    ax.set_xlabel("num_range (n)")
    ax.set_ylabel("Complement Transfer MAE")
    ax.set_title("Scaling: Error vs Problem Size")
    ax.legend()

    # Add secondary x-axis showing C(n,3)
    ax2 = axes[0].twiny()
    ax2.set_xlim(axes[0].get_xlim())
    tick_nrs = num_ranges
    tick_triples = [nr * (nr - 1) * (nr - 2) // 6 for nr in tick_nrs]
    ax2.set_xticks(tick_nrs)
    ax2.set_xticklabels([str(t) for t in tick_triples])
    ax2.set_xlabel("C(n,3) distinct triples")

    plt.tight_layout()
    plt.savefig(output_dir / "scaling_curve.png", dpi=150)
    plt.close()
    print(f"\nPlot saved to {output_dir}/scaling_curve.png")


if __name__ == "__main__":
    main()
