"""Training script for multi-group routing experiment (S_2 + S_3).

Tests whether the router can discriminate between symmetry groups:
- S_3-invariant op → should route to S_3 expert
- S_2-invariant op → should route to S_2 expert
- Non-symmetric op → should route to pass-through
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

from src.data.multigroup import MultiGroupDataset, OP_S3, OP_S2, OP_NONE
from src.models.multigroup import MultiGroupMoE, MultiGroupBaseline

OP_NAMES = {OP_S3: "S3", OP_S2: "S2", OP_NONE: "none"}


def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
    return {k: torch.stack([item[k] for item in batch]) for k in batch[0]}


def balance_loss(decision, n_options: int) -> torch.Tensor:
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
    train_s3_triples: set | None = None,
    train_s2_triples: set | None = None,
) -> dict:
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
    a_all = torch.cat(all_a)
    b_all = torch.cat(all_b)
    c_all = torch.cat(all_c)

    errors = torch.abs(preds - targets)
    results = {
        "loss": total_loss / len(preds),
        "accuracy": (errors < 0.5).float().mean().item(),
        "mae": errors.mean().item(),
    }

    # Per-op accuracy
    for op_val, op_name in OP_NAMES.items():
        mask = ops == op_val
        if mask.any():
            results[f"{op_name}_accuracy"] = (errors[mask] < 0.5).float().mean().item()
            results[f"{op_name}_mae"] = errors[mask].mean().item()

    # S_3 complement accuracy
    s3_mask = ops == OP_S3
    if train_s3_triples is not None and s3_mask.any():
        has_comp = torch.tensor([
            any((p[0], p[1], p[2]) in train_s3_triples
                for p in itertools.permutations([a_all[i].item(), b_all[i].item(), c_all[i].item()]))
            for i in range(len(a_all))
        ])
        comp_mask = has_comp & s3_mask
        if comp_mask.any():
            results["S3_comp_acc"] = (errors[comp_mask] < 0.5).float().mean().item()
            results["n_S3_comp"] = int(comp_mask.sum().item())

    # S_2 complement accuracy
    s2_mask = ops == OP_S2
    if train_s2_triples is not None and s2_mask.any():
        has_comp = torch.tensor([
            (b_all[i].item(), a_all[i].item(), c_all[i].item()) in train_s2_triples
            for i in range(len(a_all))
        ])
        comp_mask = has_comp & s2_mask
        if comp_mask.any():
            results["S2_comp_acc"] = (errors[comp_mask] < 0.5).float().mean().item()
            results["n_S2_comp"] = int(comp_mask.sum().item())

    # Routing analysis
    if routing_group_idx:
        g_idx = torch.cat(routing_group_idx)
        # group_idx: 0=pass-through, 1=S_2, 2=S_3
        for op_val, op_name in OP_NAMES.items():
            mask = ops == op_val
            if mask.any():
                results[f"{op_name}_pass_rate"] = (g_idx[mask] == 0).float().mean().item()
                results[f"{op_name}_s2_rate"] = (g_idx[mask] == 1).float().mean().item()
                results[f"{op_name}_s3_rate"] = (g_idx[mask] == 2).float().mean().item()

    return results


def train_model(model_type: str, args, device):
    print(f"\n{'='*60}")
    print(f"Training: {model_type}")
    print(f"{'='*60}")

    ds_kwargs = dict(num_range=args.num_range, seed=args.seed, train_frac=args.train_frac)
    train_ds = MultiGroupDataset(split="train", **ds_kwargs)
    stats = train_ds.get_stats()
    val_ds = MultiGroupDataset(split="val", **ds_kwargs, stats=stats)
    test_ds = MultiGroupDataset(split="test", **ds_kwargs, stats=stats)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, collate_fn=collate)

    train_s3 = train_ds.triples_for_op(OP_S3)
    train_s2 = train_ds.triples_for_op(OP_S2)

    # Count per-op examples
    for op_val, op_name in OP_NAMES.items():
        n_train = sum(1 for _, o, _, _, _ in train_ds.examples if o == op_val)
        n_test = sum(1 for _, o, _, _, _ in test_ds.examples if o == op_val)
        print(f"  {op_name}: {n_train} train, {n_test} test")

    model_kwargs = dict(d_model=args.d_model, n_numbers=args.num_range, n_blocks=args.n_blocks)
    if model_type == "groupmoe":
        model = MultiGroupMoE(**model_kwargs).to(device)
    else:
        model = MultiGroupBaseline(**model_kwargs).to(device)

    print(f"Parameters: {model.count_params():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    use_balance = model_type == "groupmoe" and args.balance_alpha > 0
    if use_balance:
        n_opts = model.group_moe.router.n_options
        print(f"Balance loss: alpha={args.balance_alpha}, n_options={n_opts}")

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

            epoch_loss += F.mse_loss(pred, target).item() * len(a)
            n_samples += len(a)

        train_loss = epoch_loss / n_samples
        val_results = evaluate(model, val_loader, train_ds.denormalize, device, train_s3, train_s2)
        test_results = evaluate(model, test_loader, train_ds.denormalize, device, train_s3, train_s2)

        history["train"].append({"epoch": epoch, "loss": train_loss})
        history["val"].append({"epoch": epoch, **val_results})
        history["test"].append({"epoch": epoch, **test_results})

        if epoch % args.log_every == 0 or epoch == 1:
            s3c = test_results.get("S3_comp_acc", 0)
            s2c = test_results.get("S2_comp_acc", 0)
            line = (
                f"[{model_type}] Epoch {epoch:3d} | "
                f"loss={train_loss:.5f} | "
                f"S3comp={s3c:.3f} S2comp={s2c:.3f}"
            )
            if "S3_s3_rate" in test_results:
                line += (
                    f" | route: S3op→S3={test_results.get('S3_s3_rate',0):.2f}"
                    f" S2op→S2={test_results.get('S2_s2_rate',0):.2f}"
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
    for op_name in ["S3", "S2", "none"]:
        acc = final.get(f"{op_name}_accuracy", 0)
        mae = final.get(f"{op_name}_mae", 0)
        print(f"  {op_name:4s} accuracy: {acc:.4f} (MAE {mae:.2f})")
    if "S3_comp_acc" in final:
        print(f"  S3 complement: {final['S3_comp_acc']:.4f} (n={final['n_S3_comp']})")
    if "S2_comp_acc" in final:
        print(f"  S2 complement: {final['S2_comp_acc']:.4f} (n={final['n_S2_comp']})")
    if "S3_s3_rate" in final:
        print(f"  Routing table:")
        print(f"    S3 op → pass={final.get('S3_pass_rate',0):.3f} S2={final.get('S3_s2_rate',0):.3f} S3={final.get('S3_s3_rate',0):.3f}")
        print(f"    S2 op → pass={final.get('S2_pass_rate',0):.3f} S2={final.get('S2_s2_rate',0):.3f} S3={final.get('S2_s3_rate',0):.3f}")
        print(f"    none  → pass={final.get('none_pass_rate',0):.3f} S2={final.get('none_s2_rate',0):.3f} S3={final.get('none_s3_rate',0):.3f}")

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
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Training Loss")
    ax.legend(); ax.set_yscale("log")

    # Per-op complement accuracy
    ax = axes[0, 1]
    for name, hist in results.items():
        epochs = [h["epoch"] for h in hist["test"]]
        s3c = [h.get("S3_comp_acc", 0) for h in hist["test"]]
        s2c = [h.get("S2_comp_acc", 0) for h in hist["test"]]
        ax.plot(epochs, s3c, label=f"{name} S3 comp", linestyle="-")
        ax.plot(epochs, s2c, label=f"{name} S2 comp", linestyle="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.set_title("Complement Transfer by Op"); ax.legend(fontsize=8)
    ax.set_ylim(-0.05, 1.05)

    # Router discrimination (GroupMoE only)
    ax = axes[1, 0]
    if "groupmoe" in results:
        hist = results["groupmoe"]
        epochs = [h["epoch"] for h in hist["test"]]
        s3_to_s3 = [h.get("S3_s3_rate", 0) for h in hist["test"]]
        s2_to_s2 = [h.get("S2_s2_rate", 0) for h in hist["test"]]
        none_pass = [h.get("none_pass_rate", 0) for h in hist["test"]]
        ax.plot(epochs, s3_to_s3, label="S3 op → S3 expert", color="blue")
        ax.plot(epochs, s2_to_s2, label="S2 op → S2 expert", color="green")
        ax.plot(epochs, none_pass, label="none op → pass-through", color="red")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Fraction")
        ax.set_title("Router: Correct Group Rate"); ax.legend()
        ax.set_ylim(-0.05, 1.05)

    # Full routing table evolution
    ax = axes[1, 1]
    if "groupmoe" in results:
        hist = results["groupmoe"]
        epochs = [h["epoch"] for h in hist["test"]]
        for op_name, color in [("S3", "blue"), ("S2", "green"), ("none", "red")]:
            to_s3 = [h.get(f"{op_name}_s3_rate", 0) for h in hist["test"]]
            to_s2 = [h.get(f"{op_name}_s2_rate", 0) for h in hist["test"]]
            ax.plot(epochs, to_s3, label=f"{op_name}→S3", color=color, linestyle="-")
            ax.plot(epochs, to_s2, label=f"{op_name}→S2", color=color, linestyle="--")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Fraction")
        ax.set_title("Full Routing Table"); ax.legend(fontsize=7)
        ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=150)
    plt.close()
    print(f"\nPlots saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Train multi-group routing (S_2 + S_3)")
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
    parser.add_argument("--output-dir", type=str, default="data/multigroup_results")
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

    models_to_train = ["groupmoe", "baseline"] if args.model == "both" else [args.model]
    all_results = {}
    for mt in models_to_train:
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
            s3c = f.get("S3_comp_acc", 0)
            s2c = f.get("S2_comp_acc", 0)
            print(f"  {name:12s}: S3comp={s3c:.3f} S2comp={s2c:.3f} overall={f['accuracy']:.3f} mae={f['mae']:.2f}")


if __name__ == "__main__":
    main()
