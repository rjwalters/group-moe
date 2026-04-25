"""Training script for arithmetic Group-MoE experiment.

Usage:
    uv run python scripts/train_arithmetic.py --model both
    uv run python scripts/train_arithmetic.py --model groupmoe --train-frac 0.3

Tests sample efficiency: with both orderings of addition in the pool,
the S_2 expert should help generalize from (a,+,b) to (b,+,a).

Key metric: "complement accuracy" — for test addition examples where
the reversed pair was in training, does GroupMoE outperform baseline?
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

from src.data.arithmetic import ArithmeticDataset, OP_ADD, OP_SUB
from src.models.arithmetic import ArithmeticGroupMoE, ArithmeticBaseline


def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
    return {k: torch.stack([item[k] for item in batch]) for k in batch[0]}


def balance_loss(decision, n_options: int) -> torch.Tensor:
    """Switch Transformer style load-balancing auxiliary loss."""
    logits = decision.logits
    probs = F.softmax(logits, dim=-1)
    best = logits.argmax(dim=-1)
    one_hot = F.one_hot(best, n_options).float()
    f = one_hot.mean(dim=0)
    p = probs.mean(dim=0)
    return n_options * (f * p).sum()


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    denormalize_fn,
    device: torch.device,
    complement_set: set[tuple[int, int]] | None = None,
) -> dict:
    """Evaluate model and compute accuracy metrics.

    If complement_set is provided, also computes complement accuracy:
    accuracy on addition test examples whose reversed pair (b,a) is in
    the complement_set (typically the training set's addition pairs).
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets_raw = []
    all_ops = []
    all_a = []
    all_b = []
    routing_group_idx = []
    routing_element_idx = []
    routing_confidence = []

    with torch.no_grad():
        for batch in loader:
            a = batch["a"].to(device)
            op = batch["op"].to(device)
            b = batch["b"].to(device)
            target = batch["target"].to(device)

            pred, decision = model(a, op, b)
            loss = F.mse_loss(pred, target)
            total_loss += loss.item() * len(a)

            pred_raw = denormalize_fn(pred.cpu())
            all_preds.append(pred_raw)
            all_targets_raw.append(batch["target_raw"])
            all_ops.append(batch["op"])
            all_a.append(batch["a"])
            all_b.append(batch["b"])

            if decision is not None:
                routing_group_idx.append(decision.group_idx.cpu())
                routing_element_idx.append(decision.element_idx.cpu())
                routing_confidence.append(decision.confidence.cpu())

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets_raw)
    ops = torch.cat(all_ops)
    a_all = torch.cat(all_a)
    b_all = torch.cat(all_b)

    errors = torch.abs(preds - targets)
    accuracy = (errors < 0.5).float().mean().item()
    mae = errors.mean().item()

    results = {
        "loss": total_loss / len(preds),
        "accuracy": accuracy,
        "mae": mae,
    }

    # Per-operation accuracy
    add_mask = ops == OP_ADD
    sub_mask = ops == OP_SUB
    if add_mask.any():
        results["add_accuracy"] = (errors[add_mask] < 0.5).float().mean().item()
        results["add_mae"] = errors[add_mask].mean().item()
    if sub_mask.any():
        results["sub_accuracy"] = (errors[sub_mask] < 0.5).float().mean().item()
        results["sub_mae"] = errors[sub_mask].mean().item()

    # Complement analysis: for addition test examples, does having
    # the reversed pair in training help?
    if complement_set is not None and add_mask.any():
        has_complement = torch.tensor([
            (b_all[i].item(), a_all[i].item()) in complement_set
            for i in range(len(a_all))
        ])
        # Only look at addition examples
        add_has_comp = has_complement & add_mask
        add_no_comp = (~has_complement) & add_mask

        if add_has_comp.any():
            results["add_with_complement_acc"] = (errors[add_has_comp] < 0.5).float().mean().item()
            results["add_with_complement_mae"] = errors[add_has_comp].mean().item()
            results["n_with_complement"] = int(add_has_comp.sum().item())
        if add_no_comp.any():
            results["add_without_complement_acc"] = (errors[add_no_comp] < 0.5).float().mean().item()
            results["add_without_complement_mae"] = errors[add_no_comp].mean().item()
            results["n_without_complement"] = int(add_no_comp.sum().item())

    # Routing analysis
    if routing_group_idx:
        g_idx = torch.cat(routing_group_idx)
        e_idx = torch.cat(routing_element_idx)
        conf = torch.cat(routing_confidence)

        results["pass_through_rate"] = (g_idx == 0).float().mean().item()
        results["s2_rate"] = (g_idx == 1).float().mean().item()
        results["mean_confidence"] = conf.mean().item()

        if add_mask.any():
            results["add_s2_rate"] = (g_idx[add_mask] == 1).float().mean().item()
        if sub_mask.any():
            results["sub_s2_rate"] = (g_idx[sub_mask] == 1).float().mean().item()

    return results


def train_model(
    model_type: str,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    print(f"\n{'='*60}")
    print(f"Training: {model_type}")
    print(f"{'='*60}")

    ds_kwargs = dict(
        num_range=args.num_range, seed=args.seed,
        train_frac=args.train_frac, split_mode=args.split_mode,
    )
    train_ds = ArithmeticDataset(split="train", **ds_kwargs)
    stats = train_ds.get_stats()
    val_ds = ArithmeticDataset(split="val", **ds_kwargs, stats=stats)
    test_ds = ArithmeticDataset(split="test", **ds_kwargs, stats=stats)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, collate_fn=collate)

    # Complement set: addition pairs in training
    train_add_pairs = train_ds.addition_pairs()
    n_add_test = sum(1 for _, op, _, _ in test_ds.examples if op == OP_ADD)
    n_with_comp = sum(
        1 for a, op, b, _ in test_ds.examples
        if op == OP_ADD and (b, a) in train_add_pairs
    )
    print(f"Data: {len(train_ds)} train, {len(val_ds)} val, {len(test_ds)} test")
    print(f"Test addition examples: {n_add_test}, with complement in train: {n_with_comp} ({n_with_comp/max(n_add_test,1)*100:.0f}%)")

    model_kwargs = dict(d_model=args.d_model, n_numbers=args.num_range, n_blocks=args.n_blocks)
    if model_type == "groupmoe":
        model = ArithmeticGroupMoE(**model_kwargs).to(device)
    else:
        model = ArithmeticBaseline(**model_kwargs).to(device)

    print(f"Parameters: {model.count_params():,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-6
    )

    use_balance = model_type == "groupmoe" and args.balance_alpha > 0
    if use_balance:
        n_router_options = model.group_moe.router.n_options
        print(f"Balance loss: alpha={args.balance_alpha}")

    history = {"train": [], "val": [], "test": []}
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_samples = 0

        for batch in train_loader:
            a = batch["a"].to(device)
            op = batch["op"].to(device)
            b = batch["b"].to(device)
            target = batch["target"].to(device)

            pred, decision = model(a, op, b)
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

        val_results = evaluate(model, val_loader, train_ds.denormalize, device, train_add_pairs)
        test_results = evaluate(model, test_loader, train_ds.denormalize, device, train_add_pairs)

        history["train"].append({"epoch": epoch, "loss": train_loss})
        history["val"].append({"epoch": epoch, **val_results})
        history["test"].append({"epoch": epoch, **test_results})

        if epoch % args.log_every == 0 or epoch == 1:
            comp_acc = test_results.get("add_with_complement_acc", 0)
            no_comp_acc = test_results.get("add_without_complement_acc", 0)
            line = (
                f"[{model_type}] Epoch {epoch:3d} | "
                f"loss={train_loss:.5f} | "
                f"test={test_results['accuracy']:.3f} | "
                f"+comp={comp_acc:.3f} -comp={no_comp_acc:.3f}"
            )
            if "s2_rate" in test_results:
                line += (
                    f" | S2: +={test_results.get('add_s2_rate', 0):.2f}"
                    f" -={test_results.get('sub_s2_rate', 0):.2f}"
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
    print(f"  Overall accuracy:  {final['accuracy']:.4f} (MAE: {final['mae']:.2f})")
    print(f"  Addition accuracy: {final.get('add_accuracy', 0):.4f} (MAE: {final.get('add_mae', 0):.2f})")
    print(f"  Subtract accuracy: {final.get('sub_accuracy', 0):.4f} (MAE: {final.get('sub_mae', 0):.2f})")
    if "add_with_complement_acc" in final:
        print(f"  + with complement:    {final['add_with_complement_acc']:.4f} "
              f"(n={final['n_with_complement']}, MAE={final.get('add_with_complement_mae',0):.2f})")
    if "add_without_complement_acc" in final:
        print(f"  + without complement: {final['add_without_complement_acc']:.4f} "
              f"(n={final['n_without_complement']}, MAE={final.get('add_without_complement_mae',0):.2f})")
    if "s2_rate" in final:
        print(f"  Routing: S2 on +={final.get('add_s2_rate', 0):.3f}, S2 on -={final.get('sub_s2_rate', 0):.3f}")

    return history


def plot_results(results: dict[str, dict], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Training loss
    ax = axes[0, 0]
    for name, hist in results.items():
        epochs = [h["epoch"] for h in hist["train"]]
        losses = [h["loss"] for h in hist["train"]]
        ax.plot(epochs, losses, label=name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss (MSE)")
    ax.set_title("Training Loss")
    ax.legend()
    ax.set_yscale("log")

    # Test accuracy
    ax = axes[0, 1]
    for name, hist in results.items():
        epochs = [h["epoch"] for h in hist["test"]]
        accs = [h["accuracy"] for h in hist["test"]]
        ax.plot(epochs, accs, label=name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Test Accuracy (overall)")
    ax.legend()
    ax.set_ylim(-0.05, 1.05)

    # Complement accuracy (the key metric)
    ax = axes[1, 0]
    for name, hist in results.items():
        epochs = [h["epoch"] for h in hist["test"]]
        comp_acc = [h.get("add_with_complement_acc", 0) for h in hist["test"]]
        no_comp_acc = [h.get("add_without_complement_acc", 0) for h in hist["test"]]
        ax.plot(epochs, comp_acc, label=f"{name} (complement in train)", linestyle="-")
        ax.plot(epochs, no_comp_acc, label=f"{name} (no complement)", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Addition Accuracy: with vs without complement")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.05, 1.05)

    # Routing evolution
    ax = axes[1, 1]
    if "groupmoe" in results:
        hist = results["groupmoe"]
        epochs = [h["epoch"] for h in hist["test"]]
        add_s2 = [h.get("add_s2_rate", 0) for h in hist["test"]]
        sub_s2 = [h.get("sub_s2_rate", 0) for h in hist["test"]]
        ax.plot(epochs, add_s2, label="+ routed to S2", color="blue")
        ax.plot(epochs, sub_s2, label="- routed to S2", color="red")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Fraction routed to S2")
        ax.set_title("Router Behavior by Operation")
        ax.legend()
        ax.set_ylim(-0.05, 1.05)
    else:
        ax.text(0.5, 0.5, "(GroupMoE not trained)", ha="center", va="center",
                transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=150)
    plt.close()
    print(f"\nPlots saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Train arithmetic Group-MoE")
    parser.add_argument("--model", choices=["groupmoe", "baseline", "both"], default="both")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-blocks", type=int, default=2)
    parser.add_argument("--num-range", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--train-frac", type=float, default=0.4)
    parser.add_argument("--balance-alpha", type=float, default=0.01)
    parser.add_argument("--split-mode", choices=["random", "complement"], default="random")
    parser.add_argument("--output-dir", type=str, default="data/arithmetic_results")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    torch.manual_seed(args.seed)

    models_to_train = (
        ["groupmoe", "baseline"] if args.model == "both" else [args.model]
    )

    all_results = {}
    for model_type in models_to_train:
        history = train_model(model_type, args, device)
        all_results[model_type] = history

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
            final = hist["test"][-1]
            comp = final.get("add_with_complement_acc", 0)
            no_comp = final.get("add_without_complement_acc", 0)
            print(f"  {name:12s}: test={final['accuracy']:.3f}  +comp={comp:.3f}  +no_comp={no_comp:.3f}  mae={final['mae']:.2f}")


if __name__ == "__main__":
    main()
