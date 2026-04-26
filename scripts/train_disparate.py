"""Training script for disparate-groups experiment (Z_2 + Z_3).

Tests whether the router can discriminate between NON-NESTED groups:
- Z_2 op (a+b+2c, swap-invariant) → should route to Z_2 expert
- Z_3 op (a²(b-c)+b²(c-a)+c²(a-b), cyclic-invariant) → should route to Z_3 expert
- Non-symmetric op (2a-b+c) → should route to pass-through
"""

from __future__ import annotations

import argparse
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

from src.data.disparate import DisparateDataset, OP_Z2, OP_Z3, OP_NONE
from src.models.disparate import DisparateGroupMoE, DisparateBaseline

OP_NAMES = {OP_Z2: "Z2", OP_Z3: "Z3", OP_NONE: "none"}


def collate(batch):
    return {k: torch.stack([item[k] for item in batch]) for k in batch[0]}


def balance_loss(decision, n_options):
    logits = decision.logits
    probs = F.softmax(logits, dim=-1)
    best = logits.argmax(dim=-1)
    one_hot = F.one_hot(best, n_options).float()
    return n_options * (one_hot.mean(0) * probs.mean(0)).sum()


def evaluate(model, loader, denormalize_fn, device, train_z2, train_z3):
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
                routing_group_idx.append(decision.group_idx.cpu())

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

    for op_val, op_name in OP_NAMES.items():
        mask = ops == op_val
        if mask.any():
            results[f"{op_name}_accuracy"] = (errors[mask] < 0.5).float().mean().item()
            results[f"{op_name}_mae"] = errors[mask].mean().item()

    # Z_2 complement (swap)
    z2_mask = ops == OP_Z2
    if train_z2 is not None and z2_mask.any():
        has_comp = torch.tensor([
            (b_all[i].item(), a_all[i].item(), c_all[i].item()) in train_z2
            for i in range(len(a_all))
        ])
        comp = has_comp & z2_mask
        if comp.any():
            results["Z2_comp_acc"] = (errors[comp] < 0.5).float().mean().item()
            results["n_Z2_comp"] = int(comp.sum().item())

    # Z_3 complement (cyclic rotation)
    z3_mask = ops == OP_Z3
    if train_z3 is not None and z3_mask.any():
        has_comp = torch.tensor([
            any(t in train_z3 for t in [
                (b_all[i].item(), c_all[i].item(), a_all[i].item()),
                (c_all[i].item(), a_all[i].item(), b_all[i].item()),
                (a_all[i].item(), b_all[i].item(), c_all[i].item()),
            ])
            for i in range(len(a_all))
        ])
        comp = has_comp & z3_mask
        if comp.any():
            results["Z3_comp_acc"] = (errors[comp] < 0.5).float().mean().item()
            results["n_Z3_comp"] = int(comp.sum().item())

    # Routing: group_idx 0=pass, 1=Z_2, 2=Z_3
    if routing_group_idx:
        g_idx = torch.cat(routing_group_idx)
        for op_val, op_name in OP_NAMES.items():
            mask = ops == op_val
            if mask.any():
                results[f"{op_name}_pass_rate"] = (g_idx[mask] == 0).float().mean().item()
                results[f"{op_name}_z2_rate"] = (g_idx[mask] == 1).float().mean().item()
                results[f"{op_name}_z3_rate"] = (g_idx[mask] == 2).float().mean().item()

    return results


def train_model(model_type, args, device):
    print(f"\n{'='*60}")
    print(f"Training: {model_type}")
    print(f"{'='*60}")

    ds_kwargs = dict(num_range=args.num_range, seed=args.seed, train_frac=args.train_frac)
    train_ds = DisparateDataset(split="train", **ds_kwargs)
    stats = train_ds.get_stats()
    val_ds = DisparateDataset(split="val", **ds_kwargs, stats=stats)
    test_ds = DisparateDataset(split="test", **ds_kwargs, stats=stats)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, collate_fn=collate)

    train_z2 = train_ds.triples_for_op(OP_Z2)
    train_z3 = train_ds.triples_for_op(OP_Z3)

    for op_val, op_name in OP_NAMES.items():
        n_train = sum(1 for _, o, _, _, _ in train_ds.examples if o == op_val)
        n_test = sum(1 for _, o, _, _, _ in test_ds.examples if o == op_val)
        print(f"  {op_name}: {n_train} train, {n_test} test")

    model_kwargs = dict(d_model=args.d_model, n_numbers=args.num_range, n_blocks=args.n_blocks)
    model = (DisparateGroupMoE(**model_kwargs) if model_type == "groupmoe"
             else DisparateBaseline(**model_kwargs)).to(device)
    print(f"Parameters: {model.count_params():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    use_balance = model_type == "groupmoe" and args.balance_alpha > 0
    n_opts = model.group_moe.router.n_options if use_balance else 0
    if use_balance:
        print(f"Balance loss: alpha={args.balance_alpha}, n_options={n_opts}")

    history = {"train": [], "val": [], "test": []}
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss, n_samples = 0.0, 0
        for batch in train_loader:
            a, op, b, c = batch["a"].to(device), batch["op"].to(device), batch["b"].to(device), batch["c"].to(device)
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
            epoch_loss += F.mse_loss(pred, target).item() * len(a)
            n_samples += len(a)

        train_loss = epoch_loss / n_samples
        val_results = evaluate(model, val_loader, train_ds.denormalize, device, train_z2, train_z3)
        test_results = evaluate(model, test_loader, train_ds.denormalize, device, train_z2, train_z3)

        history["train"].append({"epoch": epoch, "loss": train_loss})
        history["val"].append({"epoch": epoch, **val_results})
        history["test"].append({"epoch": epoch, **test_results})

        if epoch % args.log_every == 0 or epoch == 1:
            z2c = test_results.get("Z2_comp_acc", 0)
            z3c = test_results.get("Z3_comp_acc", 0)
            line = f"[{model_type}] Epoch {epoch:3d} | loss={train_loss:.5f} | Z2comp={z2c:.3f} Z3comp={z3c:.3f}"
            if "Z2_z2_rate" in test_results:
                line += (
                    f" | Z2op→Z2={test_results.get('Z2_z2_rate',0):.2f}"
                    f" Z3op→Z3={test_results.get('Z3_z3_rate',0):.2f}"
                    f" none→pass={test_results.get('none_pass_rate',0):.2f}"
                )
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
    print(f"  Overall: {final['accuracy']:.4f} (MAE {final['mae']:.2f})")
    for op_name in ["Z2", "Z3", "none"]:
        print(f"  {op_name:4s}: {final.get(f'{op_name}_accuracy', 0):.4f} (MAE {final.get(f'{op_name}_mae', 0):.2f})")
    if "Z2_comp_acc" in final:
        print(f"  Z2 complement: {final['Z2_comp_acc']:.4f} (n={final['n_Z2_comp']})")
    if "Z3_comp_acc" in final:
        print(f"  Z3 complement: {final['Z3_comp_acc']:.4f} (n={final['n_Z3_comp']})")
    if "Z2_z2_rate" in final:
        print(f"  Routing table:")
        print(f"    Z2 op → pass={final.get('Z2_pass_rate',0):.3f} Z2={final.get('Z2_z2_rate',0):.3f} Z3={final.get('Z2_z3_rate',0):.3f}")
        print(f"    Z3 op → pass={final.get('Z3_pass_rate',0):.3f} Z2={final.get('Z3_z2_rate',0):.3f} Z3={final.get('Z3_z3_rate',0):.3f}")
        print(f"    none  → pass={final.get('none_pass_rate',0):.3f} Z2={final.get('none_z2_rate',0):.3f} Z3={final.get('none_z3_rate',0):.3f}")

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
        epochs = [h["epoch"] for h in hist["test"]]
        ax.plot(epochs, [h.get("Z2_comp_acc", 0) for h in hist["test"]], label=f"{name} Z2 comp", linestyle="-")
        ax.plot(epochs, [h.get("Z3_comp_acc", 0) for h in hist["test"]], label=f"{name} Z3 comp", linestyle="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.set_title("Complement Transfer by Op"); ax.legend(fontsize=8); ax.set_ylim(-0.05, 1.05)

    ax = axes[1, 0]
    if "groupmoe" in results:
        hist = results["groupmoe"]
        epochs = [h["epoch"] for h in hist["test"]]
        ax.plot(epochs, [h.get("Z2_z2_rate", 0) for h in hist["test"]], label="Z2 op → Z2 expert", color="blue")
        ax.plot(epochs, [h.get("Z3_z3_rate", 0) for h in hist["test"]], label="Z3 op → Z3 expert", color="green")
        ax.plot(epochs, [h.get("none_pass_rate", 0) for h in hist["test"]], label="none → pass-through", color="red")
        ax.set_title("Router: Correct Group Rate"); ax.legend(); ax.set_ylim(-0.05, 1.05)

    ax = axes[1, 1]
    if "groupmoe" in results:
        hist = results["groupmoe"]
        epochs = [h["epoch"] for h in hist["test"]]
        for op_name, color in [("Z2", "blue"), ("Z3", "green"), ("none", "red")]:
            ax.plot(epochs, [h.get(f"{op_name}_z2_rate", 0) for h in hist["test"]], label=f"{op_name}→Z2", color=color, linestyle="-")
            ax.plot(epochs, [h.get(f"{op_name}_z3_rate", 0) for h in hist["test"]], label=f"{op_name}→Z3", color=color, linestyle="--")
        ax.set_title("Full Routing Table"); ax.legend(fontsize=7); ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=150)
    plt.close()
    print(f"\nPlots saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Train disparate groups (Z_2 + Z_3)")
    parser.add_argument("--model", choices=["groupmoe", "baseline", "both"], default="both")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-blocks", type=int, default=2)
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
    parser.add_argument("--output-dir", type=str, default="data/disparate_results")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("mps") if torch.backends.mps.is_available() else
        torch.device("cpu"))
    print(f"Device: {device}")
    torch.manual_seed(args.seed)

    models = ["groupmoe", "baseline"] if args.model == "both" else [args.model]
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
            print(f"  {name:12s}: Z2comp={f.get('Z2_comp_acc',0):.3f} Z3comp={f.get('Z3_comp_acc',0):.3f} overall={f['accuracy']:.3f}")


if __name__ == "__main__":
    main()
