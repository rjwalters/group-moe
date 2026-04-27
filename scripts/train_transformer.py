"""Training script for transformer Group-MoE experiment.

Same S_3 ternary task but with a small transformer encoder.
Each of (a, op, b, c) is a separate token; self-attention lets them
interact before the Group-MoE layer operates per-token.

Tests that Group-MoE works as a drop-in FFN replacement in an
attention-based architecture.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.ternary import TernaryDataset, OP_SYM, OP_NONSYM
from src.models.transformer import TransformerGroupMoE, TransformerStandardMoE, TransformerBaseline

SEQ_LEN = 4  # (a, op, b, c)


def collate(batch):
    return {k: torch.stack([item[k] for item in batch]) for k in batch[0]}


def balance_loss(decision, n_options):
    logits = decision.logits
    probs = F.softmax(logits, dim=-1)
    best = logits.argmax(dim=-1)
    one_hot = F.one_hot(best, n_options).float()
    return n_options * (one_hot.mean(0) * probs.mean(0)).sum()


def evaluate(model, loader, denormalize_fn, device, complement_set=None):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets_raw, all_ops = [], [], []
    all_a, all_b, all_c = [], [], []
    routing_group_idx = []

    with torch.no_grad():
        for batch in loader:
            a = batch["a"].to(device)
            op = batch["op"].to(device)
            b = batch["b"].to(device)
            c = batch["c"].to(device)
            target = batch["target"].to(device)

            pred, decision = model(a, op, b, c)
            total_loss += F.mse_loss(pred, target).item() * len(a)

            all_preds.append(denormalize_fn(pred.cpu()))
            all_targets_raw.append(batch["target_raw"])
            all_ops.append(batch["op"])
            all_a.append(batch["a"])
            all_b.append(batch["b"])
            all_c.append(batch["c"])

            if decision is not None:
                # Decision has shape (batch*SEQ_LEN,).
                # Take the mean routing per example by reshaping.
                g_idx = decision.group_idx.cpu().reshape(-1, SEQ_LEN)
                # For each example, compute fraction of tokens routed to S_3
                routing_group_idx.append(g_idx)

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets_raw)
    ops = torch.cat(all_ops)
    a_all, b_all, c_all = torch.cat(all_a), torch.cat(all_b), torch.cat(all_c)

    errors = torch.abs(preds - targets)
    results = {
        "loss": total_loss / len(preds),
        "accuracy": (errors < 0.5).float().mean().item(),
        "mae": errors.mean().item(),
    }

    sym_mask = ops == OP_SYM
    nonsym_mask = ops == OP_NONSYM
    if sym_mask.any():
        results["sym_accuracy"] = (errors[sym_mask] < 0.5).float().mean().item()
        results["sym_mae"] = errors[sym_mask].mean().item()
    if nonsym_mask.any():
        results["nonsym_accuracy"] = (errors[nonsym_mask] < 0.5).float().mean().item()
        results["nonsym_mae"] = errors[nonsym_mask].mean().item()

    # Complement accuracy
    if complement_set is not None and sym_mask.any():
        has_complement = torch.tensor([
            any((p[0], p[1], p[2]) in complement_set
                for p in itertools.permutations([a_all[i].item(), b_all[i].item(), c_all[i].item()]))
            for i in range(len(a_all))
        ])
        comp_mask = has_complement & sym_mask
        if comp_mask.any():
            results["sym_with_complement_acc"] = (errors[comp_mask] < 0.5).float().mean().item()
            results["n_with_complement"] = int(comp_mask.sum().item())

    # Routing: per-example S_3 rate (fraction of 4 tokens routed to S_3)
    if routing_group_idx:
        g_idx = torch.cat(routing_group_idx, dim=0)  # (n_examples, SEQ_LEN)
        s3_frac = (g_idx == 1).float().mean(dim=1)  # per-example S_3 fraction

        results["s3_rate"] = s3_frac.mean().item()
        if sym_mask.any():
            results["sym_s3_rate"] = s3_frac[sym_mask].mean().item()
        if nonsym_mask.any():
            results["nonsym_s3_rate"] = s3_frac[nonsym_mask].mean().item()

    return results


def train_model(model_type, args, device):
    print(f"\n{'='*60}")
    print(f"Training: {model_type}")
    print(f"{'='*60}")

    ds_kwargs = dict(
        num_range=args.num_range, seed=args.seed,
        train_frac=args.train_frac, split_mode=args.split_mode,
    )
    train_ds = TernaryDataset(split="train", **ds_kwargs)
    stats = train_ds.get_stats()
    val_ds = TernaryDataset(split="val", **ds_kwargs, stats=stats)
    test_ds = TernaryDataset(split="test", **ds_kwargs, stats=stats)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, collate_fn=collate)

    train_sym_triples = train_ds.symmetric_triples()
    n_sym_test = sum(1 for _, op, _, _, _ in test_ds.examples if op == OP_SYM)
    print(f"Data: {len(train_ds)} train, {len(val_ds)} val, {len(test_ds)} test")
    print(f"Test symmetric examples: {n_sym_test}")

    model_kwargs = dict(
        d_model=args.d_model, n_numbers=args.num_range, n_heads=args.n_heads,
        n_layers=args.n_layers, ffn_expansion=args.ffn_expansion, dropout=args.dropout,
    )
    if model_type == "groupmoe":
        model = TransformerGroupMoE(moe_layer_idx=args.moe_layer, **model_kwargs).to(device)
    elif model_type == "standardmoe":
        model = TransformerStandardMoE(moe_layer_idx=args.moe_layer, **model_kwargs).to(device)
    else:
        model = TransformerBaseline(**model_kwargs).to(device)
    print(f"Parameters: {model.count_params():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    use_balance = model_type in ("groupmoe", "standardmoe") and args.balance_alpha > 0
    if use_balance:
        moe_block = model.blocks[args.moe_layer]
        moe_layer = moe_block.moe_layer
        n_router_options = moe_layer.n_options if hasattr(moe_layer, 'n_options') else moe_layer.router.n_options
        print(f"Balance loss: alpha={args.balance_alpha}, n_options={n_router_options}")

    history = {"train": [], "val": [], "test": []}
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss, n_samples = 0.0, 0

        for batch in train_loader:
            a = batch["a"].to(device)
            op = batch["op"].to(device)
            b = batch["b"].to(device)
            c = batch["c"].to(device)
            target = batch["target"].to(device)

            pred, decision = model(a, op, b, c)
            loss = F.mse_loss(pred, target)

            if use_balance and decision is not None:
                loss = loss + args.balance_alpha * balance_loss(decision, n_router_options)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += F.mse_loss(pred, target).item() * len(a)
            n_samples += len(a)

        train_loss = epoch_loss / n_samples
        val_results = evaluate(model, val_loader, train_ds.denormalize, device, train_sym_triples)
        test_results = evaluate(model, test_loader, train_ds.denormalize, device, train_sym_triples)

        history["train"].append({"epoch": epoch, "loss": train_loss})
        history["val"].append({"epoch": epoch, **val_results})
        history["test"].append({"epoch": epoch, **test_results})

        if epoch % args.log_every == 0 or epoch == 1:
            comp_acc = test_results.get("sym_with_complement_acc", 0)
            line = f"[{model_type}] Epoch {epoch:3d} | loss={train_loss:.5f} | test={test_results['accuracy']:.3f} | +comp={comp_acc:.3f}"
            if "s3_rate" in test_results:
                line += f" | S3: sym={test_results.get('sym_s3_rate', 0):.2f} nsym={test_results.get('nonsym_s3_rate', 0):.2f}"
            print(line)

        if val_results["loss"] < best_val_loss:
            best_val_loss = val_results["loss"]
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    final = history["test"][-1]
    print(f"\n--- {model_type} Final ---")
    print(f"  Overall:   {final['accuracy']:.4f} (MAE {final['mae']:.2f})")
    print(f"  Symmetric: {final.get('sym_accuracy', 0):.4f}")
    print(f"  Non-sym:   {final.get('nonsym_accuracy', 0):.4f}")
    if "sym_with_complement_acc" in final:
        print(f"  Complement: {final['sym_with_complement_acc']:.4f} (n={final['n_with_complement']})")
    if "sym_s3_rate" in final:
        print(f"  S3 rate: sym={final['sym_s3_rate']:.3f} nonsym={final['nonsym_s3_rate']:.3f}")

    return history


def plot_results(results, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    for name, hist in results.items():
        ax.plot([h["epoch"] for h in hist["train"]], [h["loss"] for h in hist["train"]], label=name)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Training Loss")
    ax.legend(); ax.set_yscale("log")

    ax = axes[0, 1]
    for name, hist in results.items():
        ax.plot([h["epoch"] for h in hist["test"]], [h["accuracy"] for h in hist["test"]], label=name)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy"); ax.set_title("Test Accuracy")
    ax.legend(); ax.set_ylim(-0.05, 1.05)

    ax = axes[1, 0]
    for name, hist in results.items():
        ax.plot([h["epoch"] for h in hist["test"]],
                [h.get("sym_with_complement_acc", 0) for h in hist["test"]], label=f"{name}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy"); ax.set_title("Complement Transfer (symmetric)")
    ax.legend(); ax.set_ylim(-0.05, 1.05)

    ax = axes[1, 1]
    for name, hist in results.items():
        if any("sym_s3_rate" in h for h in hist["test"]):
            ax.plot([h["epoch"] for h in hist["test"]],
                    [h.get("sym_s3_rate", 0) for h in hist["test"]], label=f"{name} sym", linestyle="-")
            ax.plot([h["epoch"] for h in hist["test"]],
                    [h.get("nonsym_s3_rate", 0) for h in hist["test"]], label=f"{name} nsym", linestyle="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("S3 rate"); ax.set_title("Router: S3 Rate by Op")
    ax.legend(fontsize=8); ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=150)
    plt.close()
    print(f"\nPlots saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Train transformer Group-MoE (S_3)")
    parser.add_argument("--model", choices=["groupmoe", "standardmoe", "baseline", "both", "all"], default="all")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--ffn-expansion", type=int, default=4)
    parser.add_argument("--moe-layer", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--num-range", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=75)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--train-frac", type=float, default=0.5)
    parser.add_argument("--balance-alpha", type=float, default=0.01)
    parser.add_argument("--split-mode", choices=["random", "complement", "composition"], default="complement")
    parser.add_argument("--output-dir", type=str, default="data/transformer_results")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("mps") if torch.backends.mps.is_available() else
        torch.device("cpu"))
    print(f"Device: {device}")
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
    plot_results(all_results, output_dir)

    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("COMPARISON")
        print(f"{'='*60}")
        for name, hist in all_results.items():
            f = hist["test"][-1]
            comp = f.get("sym_with_complement_acc", 0)
            print(f"  {name:14s}: comp={comp:.3f} overall={f['accuracy']:.3f} mae={f['mae']:.2f}")


if __name__ == "__main__":
    main()
