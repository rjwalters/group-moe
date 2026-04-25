"""Per-pair analysis of complement transfer.

Trains both models with complement split, then examines:
- Which reversed addition pairs each model gets right/wrong
- Router confidence by pair
- Error patterns (near-misses vs large errors)
"""

from __future__ import annotations

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

from src.data.arithmetic import ArithmeticDataset, OP_ADD, OP_SUB
from src.models.arithmetic import ArithmeticGroupMoE, ArithmeticBaseline


def collate(batch):
    return {k: torch.stack([item[k] for item in batch]) for k in batch[0]}


def balance_loss(decision, n_options):
    logits = decision.logits
    probs = F.softmax(logits, dim=-1)
    best = logits.argmax(dim=-1)
    one_hot = F.one_hot(best, n_options).float()
    f = one_hot.mean(dim=0)
    p = probs.mean(dim=0)
    return n_options * (f * p).sum()


def train(model, train_loader, device, epochs=300, lr=3e-4,
          weight_decay=1e-2, balance_alpha=0.01):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    use_balance = hasattr(model, 'group_moe') and balance_alpha > 0
    if use_balance:
        n_opts = model.group_moe.router.n_options

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            a = batch["a"].to(device)
            op = batch["op"].to(device)
            b = batch["b"].to(device)
            target = batch["target"].to(device)

            pred, decision = model(a, op, b)
            loss = F.mse_loss(pred, target)
            if use_balance and decision is not None:
                loss = loss + balance_alpha * balance_loss(decision, n_opts)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        if epoch % 50 == 0:
            model.eval()
            with torch.no_grad():
                total_loss = 0.0
                n = 0
                for batch in train_loader:
                    pred, _ = model(batch["a"].to(device), batch["op"].to(device), batch["b"].to(device))
                    total_loss += F.mse_loss(pred, batch["target"].to(device)).item() * len(pred)
                    n += len(pred)
            print(f"  Epoch {epoch:4d} train_loss={total_loss/n:.6f}")


def analyze(model, test_loader, denormalize_fn, device, model_name):
    """Run model on test set, return per-example results."""
    model.eval()
    results = []

    with torch.no_grad():
        for batch in test_loader:
            a = batch["a"].to(device)
            op = batch["op"].to(device)
            b = batch["b"].to(device)

            pred, decision = model(a, op, b)
            pred_raw = denormalize_fn(pred.cpu())

            for i in range(len(a)):
                row = {
                    "a": a[i].item(),
                    "op": op[i].item(),
                    "b": b[i].item(),
                    "target": batch["target_raw"][i].item(),
                    "pred": pred_raw[i].item(),
                    "error": abs(pred_raw[i].item() - batch["target_raw"][i].item()),
                    "correct": abs(pred_raw[i].item() - batch["target_raw"][i].item()) < 0.5,
                }
                if decision is not None:
                    row["group_idx"] = decision.group_idx[i].item()
                    row["element_idx"] = decision.element_idx[i].item()
                    row["confidence"] = decision.confidence[i].item()
                    row["s2_logit"] = decision.logits[i, 1].item() if decision.logits.shape[-1] > 1 else 0
                results.append(row)

    return results


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    torch.manual_seed(42)

    num_range = 20
    ds_kwargs = dict(num_range=num_range, seed=42, train_frac=0.5, split_mode="complement")
    train_ds = ArithmeticDataset(split="train", **ds_kwargs)
    stats = train_ds.get_stats()
    test_ds = ArithmeticDataset(split="test", **ds_kwargs, stats=stats)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=256, collate_fn=collate)

    train_add_pairs = train_ds.addition_pairs()
    print(f"Train: {len(train_ds)} examples, Test: {len(test_ds)} examples")
    print(f"Train addition pairs: {len(train_add_pairs)}")

    # Train GroupMoE
    print("\n--- Training GroupMoE ---")
    gmoe = ArithmeticGroupMoE(d_model=128, n_numbers=num_range, n_blocks=2).to(device)
    train(gmoe, train_loader, device, epochs=300)
    gmoe_results = analyze(gmoe, test_loader, train_ds.denormalize, device, "groupmoe")

    # Train Baseline
    print("\n--- Training Baseline ---")
    torch.manual_seed(42)
    base = ArithmeticBaseline(d_model=128, n_numbers=num_range, n_blocks=2).to(device)
    train(base, train_loader, device, epochs=300, balance_alpha=0.0)
    base_results = analyze(base, test_loader, train_ds.denormalize, device, "baseline")

    # --- Analysis ---
    add_gmoe = [r for r in gmoe_results if r["op"] == OP_ADD]
    add_base = [r for r in base_results if r["op"] == OP_ADD]
    sub_gmoe = [r for r in gmoe_results if r["op"] == OP_SUB]
    sub_base = [r for r in base_results if r["op"] == OP_SUB]

    print(f"\n{'='*60}")
    print("AGGREGATE RESULTS")
    print(f"{'='*60}")
    print(f"  GroupMoE addition: {sum(r['correct'] for r in add_gmoe)}/{len(add_gmoe)} "
          f"({sum(r['correct'] for r in add_gmoe)/len(add_gmoe)*100:.1f}%)")
    print(f"  Baseline addition: {sum(r['correct'] for r in add_base)}/{len(add_base)} "
          f"({sum(r['correct'] for r in add_base)/len(add_base)*100:.1f}%)")
    print(f"  GroupMoE subtract: {sum(r['correct'] for r in sub_gmoe)}/{len(sub_gmoe)} "
          f"({sum(r['correct'] for r in sub_gmoe)/len(sub_gmoe)*100:.1f}%)")
    print(f"  Baseline subtract: {sum(r['correct'] for r in sub_base)}/{len(sub_base)} "
          f"({sum(r['correct'] for r in sub_base)/len(sub_base)*100:.1f}%)")

    # Per-pair comparison for addition
    print(f"\n{'='*60}")
    print("PER-PAIR ANALYSIS (addition complement transfer)")
    print(f"{'='*60}")

    gmoe_by_pair = {(r["a"], r["b"]): r for r in add_gmoe}
    base_by_pair = {(r["a"], r["b"]): r for r in add_base}

    both_right = 0
    gmoe_only = []
    base_only = []
    both_wrong = []

    for pair in sorted(gmoe_by_pair.keys()):
        g = gmoe_by_pair[pair]
        b = base_by_pair.get(pair)
        if b is None:
            continue

        if g["correct"] and b["correct"]:
            both_right += 1
        elif g["correct"] and not b["correct"]:
            gmoe_only.append((pair, g, b))
        elif not g["correct"] and b["correct"]:
            base_only.append((pair, g, b))
        else:
            both_wrong.append((pair, g, b))

    total = both_right + len(gmoe_only) + len(base_only) + len(both_wrong)
    print(f"  Both correct:     {both_right}/{total} ({both_right/total*100:.1f}%)")
    print(f"  GroupMoE only:    {len(gmoe_only)}/{total} ({len(gmoe_only)/total*100:.1f}%)")
    print(f"  Baseline only:    {len(base_only)}/{total} ({len(base_only)/total*100:.1f}%)")
    print(f"  Both wrong:       {len(both_wrong)}/{total} ({len(both_wrong)/total*100:.1f}%)")

    if gmoe_only:
        print(f"\n  GroupMoE-only correct (first 10):")
        print(f"  {'pair':>10s} {'target':>7s} {'gmoe_pred':>9s} {'base_pred':>9s} {'routed_to':>10s} {'conf':>6s}")
        for pair, g, b in gmoe_only[:10]:
            route = "S2" if g.get("group_idx", 0) == 1 else "pass"
            conf = f"{g.get('confidence', 0):.3f}"
            print(f"  {str(pair):>10s} {g['target']:>7.1f} {g['pred']:>9.2f} {b['pred']:>9.2f} {route:>10s} {conf:>6s}")

    if base_only:
        print(f"\n  Baseline-only correct (first 10):")
        print(f"  {'pair':>10s} {'target':>7s} {'gmoe_pred':>9s} {'base_pred':>9s} {'routed_to':>10s}")
        for pair, g, b in base_only[:10]:
            route = "S2" if g.get("group_idx", 0) == 1 else "pass"
            print(f"  {str(pair):>10s} {g['target']:>7.1f} {g['pred']:>9.2f} {b['pred']:>9.2f} {route:>10s}")

    # Error distribution
    gmoe_add_errors = np.array([r["error"] for r in add_gmoe])
    base_add_errors = np.array([r["error"] for r in add_base])

    print(f"\n{'='*60}")
    print("ERROR DISTRIBUTION (addition)")
    print(f"{'='*60}")
    for thresh in [0.5, 1.0, 2.0, 5.0]:
        g_pct = (gmoe_add_errors < thresh).mean() * 100
        b_pct = (base_add_errors < thresh).mean() * 100
        print(f"  Error < {thresh:.1f}: GroupMoE={g_pct:.1f}%  Baseline={b_pct:.1f}%")
    print(f"  Mean error:  GroupMoE={gmoe_add_errors.mean():.3f}  Baseline={base_add_errors.mean():.3f}")
    print(f"  Median error: GroupMoE={np.median(gmoe_add_errors):.3f}  Baseline={np.median(base_add_errors):.3f}")

    # Router analysis
    print(f"\n{'='*60}")
    print("ROUTER ANALYSIS")
    print(f"{'='*60}")
    if add_gmoe[0].get("group_idx") is not None:
        # By operation
        add_s2_rate = np.mean([r["group_idx"] == 1 for r in add_gmoe])
        sub_s2_rate = np.mean([r["group_idx"] == 1 for r in sub_gmoe])
        print(f"  S2 rate: addition={add_s2_rate:.3f}  subtraction={sub_s2_rate:.3f}")

        # By correctness
        correct_s2 = np.mean([r["group_idx"] == 1 for r in add_gmoe if r["correct"]])
        wrong_s2 = np.mean([r["group_idx"] == 1 for r in add_gmoe if not r["correct"]])
        print(f"  S2 rate when correct={correct_s2:.3f}  when wrong={wrong_s2:.3f}")

        # Confidence
        correct_conf = np.mean([r["confidence"] for r in add_gmoe if r["correct"]])
        wrong_conf = np.mean([r["confidence"] for r in add_gmoe if not r["correct"]])
        print(f"  Confidence when correct={correct_conf:.3f}  when wrong={wrong_conf:.3f}")

        # S2 logit distribution
        add_s2_logits = [r["s2_logit"] for r in add_gmoe]
        sub_s2_logits = [r["s2_logit"] for r in sub_gmoe]
        print(f"  S2 logit: addition mean={np.mean(add_s2_logits):.3f} std={np.std(add_s2_logits):.3f}")
        print(f"  S2 logit: subtract mean={np.mean(sub_s2_logits):.3f} std={np.std(sub_s2_logits):.3f}")

    # --- Plots ---
    output_dir = Path("data/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Error heatmap for GroupMoE (addition test pairs)
    ax = axes[0, 0]
    err_grid = np.full((num_range, num_range), np.nan)
    for r in add_gmoe:
        err_grid[r["a"], r["b"]] = r["error"]
    im = ax.imshow(err_grid, cmap="RdYlGn_r", vmin=0, vmax=5, origin="lower")
    ax.set_xlabel("b")
    ax.set_ylabel("a")
    ax.set_title("GroupMoE: addition error (b,+,a)")
    plt.colorbar(im, ax=ax)

    # Error heatmap for Baseline
    ax = axes[0, 1]
    err_grid_b = np.full((num_range, num_range), np.nan)
    for r in add_base:
        err_grid_b[r["a"], r["b"]] = r["error"]
    im = ax.imshow(err_grid_b, cmap="RdYlGn_r", vmin=0, vmax=5, origin="lower")
    ax.set_xlabel("b")
    ax.set_ylabel("a")
    ax.set_title("Baseline: addition error (b,+,a)")
    plt.colorbar(im, ax=ax)

    # Advantage heatmap (baseline error - gmoe error)
    ax = axes[1, 0]
    adv_grid = np.full((num_range, num_range), np.nan)
    for pair in gmoe_by_pair:
        if pair in base_by_pair:
            adv_grid[pair[0], pair[1]] = base_by_pair[pair]["error"] - gmoe_by_pair[pair]["error"]
    im = ax.imshow(adv_grid, cmap="RdBu", vmin=-3, vmax=3, origin="lower")
    ax.set_xlabel("b")
    ax.set_ylabel("a")
    ax.set_title("Advantage (blue = GroupMoE better)")
    plt.colorbar(im, ax=ax)

    # Router S2 probability heatmap
    ax = axes[1, 1]
    if add_gmoe[0].get("group_idx") is not None:
        route_grid = np.full((num_range, num_range), np.nan)
        for r in add_gmoe:
            route_grid[r["a"], r["b"]] = 1 if r["group_idx"] == 1 else 0
        im = ax.imshow(route_grid, cmap="Blues", vmin=0, vmax=1, origin="lower")
        ax.set_xlabel("b")
        ax.set_ylabel("a")
        ax.set_title("Router: routed to S2 (addition)")
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(output_dir / "complement_analysis.png", dpi=150)
    plt.close()
    print(f"\nPlots saved to {output_dir}/")


if __name__ == "__main__":
    main()
