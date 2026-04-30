"""Training script for S_n scaling experiments.

Tests whether Group-MoE's advantage over learned transforms grows
with group order. Sweeps n = 3, 4, 5 (S_3, S_4, S_5) with fixed
model capacity.

Usage:
    uv run python scripts/train_nary.py --n 4 --model all
    uv run python scripts/train_nary.py --n 5 --model groupmoe --num-range 6
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import permutations as iter_perms
from math import factorial
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.nary import NaryDataset, OP_SYM, OP_NONSYM
from src.models.nary import NaryGroupMoE, NaryStandardMoE, NaryBaseline


def collate(batch):
    return {k: torch.stack([item[k] for item in batch]) for k in batch[0]}


def balance_loss(decision, n_options):
    logits = decision.logits
    probs = F.softmax(logits, dim=-1)
    best = logits.argmax(dim=-1)
    one_hot = F.one_hot(best, n_options).float()
    return n_options * (one_hot.mean(0) * probs.mean(0)).sum()


def evaluate(model, loader, denorm, device, train_sym_inputs):
    model.eval()
    all_preds, all_targets, all_ops, all_inputs = [], [], [], []
    routing_group_idx = []

    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            op = batch["op"].to(device)
            pred, decision = model(inputs, op)
            all_preds.append(denorm(pred.cpu()))
            all_targets.append(batch["target_raw"])
            all_ops.append(batch["op"])
            all_inputs.append(batch["inputs"])
            if decision is not None:
                routing_group_idx.append(decision.group_idx.cpu())

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    ops = torch.cat(all_ops)
    inputs_all = torch.cat(all_inputs)

    errors = torch.abs(preds - targets)
    sym_mask = ops == OP_SYM
    results = {
        "accuracy": (errors < 0.5).float().mean().item(),
        "mae": errors.mean().item(),
    }

    if sym_mask.any():
        results["sym_accuracy"] = (errors[sym_mask] < 0.5).float().mean().item()

    # Complement accuracy
    if train_sym_inputs is not None and sym_mask.any():
        has_comp = torch.tensor([
            any(tuple(p) in train_sym_inputs
                for p in iter_perms(inputs_all[i].tolist()))
            for i in range(len(inputs_all))
        ])
        comp = has_comp & sym_mask
        if comp.any():
            results["complement_acc"] = (errors[comp] < 0.5).float().mean().item()
            results["n_complement"] = int(comp.sum().item())

    # Routing
    if routing_group_idx:
        g_idx = torch.cat(routing_group_idx)
        results["expert_rate"] = (g_idx == 1).float().mean().item()
        if sym_mask.any():
            results["sym_expert_rate"] = (g_idx[sym_mask] == 1).float().mean().item()
        if (~sym_mask).any():
            results["nonsym_expert_rate"] = (g_idx[~sym_mask] == 1).float().mean().item()

    return results


def train_model(model_type, args, device):
    n = args.n
    print(f"\n{'='*60}")
    print(f"Training: {model_type} (S_{n}, order {factorial(n)})")
    print(f"{'='*60}")

    ds_kwargs = dict(n=n, num_range=args.num_range, seed=args.seed, train_frac=args.train_frac)
    train_ds = NaryDataset(split="train", **ds_kwargs)
    stats = train_ds.get_stats()
    val_ds = NaryDataset(split="val", **ds_kwargs, stats=stats)
    test_ds = NaryDataset(split="test", **ds_kwargs, stats=stats)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, collate_fn=collate)

    train_sym = train_ds.symmetric_inputs()
    n_sym_test = sum(1 for _, op, _ in test_ds.examples if op == OP_SYM)
    print(f"Data: {len(train_ds)} train, {len(val_ds)} val, {len(test_ds)} test")
    print(f"Test symmetric: {n_sym_test}")

    model_kwargs = dict(n=n, d_model=args.d_model, n_numbers=args.num_range, n_blocks=args.n_blocks)
    if model_type == "groupmoe":
        model = NaryGroupMoE(**model_kwargs).to(device)
    elif model_type == "standardmoe":
        model = NaryStandardMoE(**model_kwargs).to(device)
    else:
        model = NaryBaseline(**model_kwargs).to(device)
    print(f"Parameters: {model.count_params():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    use_balance = model_type in ("groupmoe", "standardmoe") and args.balance_alpha > 0
    if use_balance:
        moe = model.group_moe if hasattr(model, 'group_moe') else model.standard_moe
        n_opts = moe.n_options if hasattr(moe, 'n_options') else moe.router.n_options
        print(f"Balance loss: alpha={args.balance_alpha}, n_options={n_opts}")

    history = {"train": [], "test": []}
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss, n_samples = 0.0, 0
        for batch in train_loader:
            inputs = batch["inputs"].to(device)
            op = batch["op"].to(device)
            target = batch["target"].to(device)
            pred, decision = model(inputs, op)
            loss = F.mse_loss(pred, target)
            if use_balance and decision is not None:
                loss = loss + args.balance_alpha * balance_loss(decision, n_opts)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += F.mse_loss(pred, target).item() * len(inputs)
            n_samples += len(inputs)

        train_loss = epoch_loss / n_samples

        val_r = evaluate(model, val_loader, train_ds.denormalize, device, train_sym)
        test_r = evaluate(model, test_loader, train_ds.denormalize, device, train_sym)

        history["train"].append({"epoch": epoch, "loss": train_loss})
        history["test"].append({"epoch": epoch, **test_r})

        if epoch % args.log_every == 0 or epoch == 1:
            comp = test_r.get("complement_acc", 0)
            line = f"[{model_type}] Epoch {epoch:3d} | loss={train_loss:.5f} | comp={comp:.3f}"
            if "sym_expert_rate" in test_r:
                line += f" | expert: sym={test_r['sym_expert_rate']:.2f} nsym={test_r.get('nonsym_expert_rate',0):.2f}"
            print(line)

        val_loss = val_r.get("mae", float("inf"))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    final = history["test"][-1]
    print(f"\n--- {model_type} Final ---")
    print(f"  Overall: {final['accuracy']:.4f} (MAE {final['mae']:.2f})")
    print(f"  Symmetric: {final.get('sym_accuracy', 0):.4f}")
    if "complement_acc" in final:
        print(f"  Complement: {final['complement_acc']:.4f} (n={final['n_complement']})")

    return history


def main():
    parser = argparse.ArgumentParser(description="Train n-ary Group-MoE (S_n scaling)")
    parser.add_argument("--n", type=int, default=4, help="Arity (S_n group order = n!)")
    parser.add_argument("--model", choices=["groupmoe", "standardmoe", "baseline", "both", "all"], default="all")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-blocks", type=int, default=2)
    parser.add_argument("--num-range", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=75)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--train-frac", type=float, default=0.5)
    parser.add_argument("--balance-alpha", type=float, default=0.01)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"data/nary_s{args.n}"

    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("mps") if torch.backends.mps.is_available() else
        torch.device("cpu"))
    print(f"Device: {device}")
    print(f"Group: S_{args.n} (order {factorial(args.n)}, irrep total_dim={_get_total_dim(args.n)})")
    torch.manual_seed(args.seed)

    if args.model == "all":
        models = ["groupmoe", "standardmoe", "baseline"]
    elif args.model == "both":
        models = ["groupmoe", "baseline"]
    else:
        models = [args.model]

    all_results = {}
    for mt in models:
        all_results[mt] = train_model(mt, args, device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print(f"COMPARISON (S_{args.n})")
        print(f"{'='*60}")
        for name, hist in all_results.items():
            f = hist["test"][-1]
            comp = f.get("complement_acc", 0)
            print(f"  {name:14s}: comp={comp:.3f} overall={f['accuracy']:.3f} mae={f['mae']:.2f}")


def _get_total_dim(n):
    from src.groups.symmetric import partitions, hook_length_dim
    return sum(hook_length_dim(p) for p in partitions(n))


if __name__ == "__main__":
    main()
