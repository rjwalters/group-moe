"""Coverage sweep: vary the fraction of orderings seen in training.

For each distinct triple {a,b,c}, put k orderings in training and
6-k in test (k = 1, 2, 3, 4). This varies the "LUT coverage" from
17% to 67%. The group expert's advantage should grow as coverage drops
because the model must generalize from fewer examples.

k=1: 17% coverage — maximum generalization challenge (our complement split)
k=2: 33% coverage
k=3: 50% coverage
k=4: 67% coverage — most of the table seen, little generalization needed
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
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.ternary import OP_SYM, OP_NONSYM, symmetric_fn, nonsym_fn
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


class CoverageDataset(Dataset):
    """Ternary dataset with configurable ordering coverage.

    For each distinct triple, k_train orderings go to training,
    remaining to test. Non-symmetric op split randomly.
    """

    def __init__(
        self, split, num_range=15, seed=42, k_train=1,
        train_frac=0.5, stats=None,
    ):
        self.num_range = num_range
        rng = np.random.RandomState(seed)

        sym_train, sym_test = [], []

        # Distinct triples
        for a in range(num_range):
            for b in range(a + 1, num_range):
                for c in range(b + 1, num_range):
                    orderings = list(itertools.permutations([a, b, c]))
                    rng.shuffle(orderings)
                    val = symmetric_fn(a, b, c)
                    for o in orderings[:k_train]:
                        sym_train.append((o[0], OP_SYM, o[1], o[2], val))
                    for o in orderings[k_train:]:
                        sym_test.append((o[0], OP_SYM, o[1], o[2], val))

        # Two-equal: scale k_train proportionally (max 3 orderings)
        k_two = max(1, min(k_train, 2))
        for a in range(num_range):
            for b in range(num_range):
                if a == b:
                    continue
                orderings = [(a, a, b), (a, b, a), (b, a, a)]
                rng.shuffle(orderings)
                val = symmetric_fn(a, a, b)
                for o in orderings[:k_two]:
                    sym_train.append((o[0], OP_SYM, o[1], o[2], val))
                for o in orderings[k_two:]:
                    sym_test.append((o[0], OP_SYM, o[1], o[2], val))

        # All-equal: train only
        for a in range(num_range):
            sym_train.append((a, OP_SYM, a, a, symmetric_fn(a, a, a)))

        # Non-symmetric: random split
        nonsym_all = []
        for a in range(num_range):
            for b in range(num_range):
                for c in range(num_range):
                    nonsym_all.append((a, OP_NONSYM, b, c, nonsym_fn(a, b, c)))
        rng.shuffle(nonsym_all)
        n_train = int(len(nonsym_all) * train_frac)
        n_val = int(len(nonsym_all) * 0.1)

        if split == "train":
            self.examples = sym_train + nonsym_all[:n_train]
        elif split == "val":
            rng2 = np.random.RandomState(seed + 1)
            n_sym_val = max(1, int(len(sym_test) * 0.1))
            rng2.shuffle(sym_test)
            self.examples = sym_test[:n_sym_val] + nonsym_all[n_train:n_train + n_val]
        else:
            self.examples = sym_test + nonsym_all[n_train + n_val:]

        rng3 = np.random.RandomState(seed + 2)
        rng3.shuffle(self.examples)

        raw = np.array([ex[4] for ex in self.examples], dtype=np.float64)
        if stats:
            self.mean, self.std = stats
        else:
            self.mean = float(raw.mean())
            self.std = float(raw.std()) if len(raw) > 1 else 1.0

    def get_stats(self):
        return (self.mean, self.std)

    def denormalize(self, t):
        return t * self.std + self.mean

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        a, op, b, c, result = self.examples[idx]
        target_raw = float(result)
        target = (target_raw - self.mean) / self.std
        return {
            "a": torch.tensor(a, dtype=torch.long),
            "op": torch.tensor(op, dtype=torch.long),
            "b": torch.tensor(b, dtype=torch.long),
            "c": torch.tensor(c, dtype=torch.long),
            "target": torch.tensor(target, dtype=torch.float32),
            "target_raw": torch.tensor(target_raw, dtype=torch.float32),
        }

    def symmetric_triples(self):
        return {(a, b, c) for a, op, b, c, _ in self.examples if op == OP_SYM}


def evaluate(model, loader, denorm, device, train_triples):
    model.eval()
    all_preds, all_targets, all_ops = [], [], []
    all_a, all_b, all_c = [], [], []
    with torch.no_grad():
        for batch in loader:
            pred, _ = model(
                batch["a"].to(device), batch["op"].to(device),
                batch["b"].to(device), batch["c"].to(device))
            all_preds.append(denorm(pred.cpu()))
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
    results = {"accuracy": (errors < 0.5).float().mean().item()}

    if sym_mask.any():
        has_comp = torch.tensor([
            any((p[0], p[1], p[2]) in train_triples
                for p in perms([a_all[i].item(), b_all[i].item(), c_all[i].item()]))
            for i in range(len(a_all))
        ])
        comp = has_comp & sym_mask
        if comp.any():
            results["complement_acc"] = (errors[comp] < 0.5).float().mean().item()
    return results


def train_and_eval(model_type, k_train, args, device):
    train_ds = CoverageDataset("train", args.num_range, args.seed, k_train, args.train_frac)
    stats = train_ds.get_stats()
    val_ds = CoverageDataset("val", args.num_range, args.seed, k_train, args.train_frac, stats)
    test_ds = CoverageDataset("test", args.num_range, args.seed, k_train, args.train_frac, stats)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, collate_fn=collate)
    train_triples = train_ds.symmetric_triples()

    model_kwargs = dict(d_model=args.d_model, n_numbers=args.num_range, n_blocks=args.n_blocks)
    if model_type == "groupmoe":
        model = TernaryGroupMoE(**model_kwargs).to(device)
    elif model_type == "standardmoe":
        model = TernaryStandardMoE(**model_kwargs).to(device)
    else:
        model = TernaryBaseline(**model_kwargs).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_loader) * args.epochs, eta_min=1e-6)

    use_balance = model_type in ("groupmoe", "standardmoe") and args.balance_alpha > 0
    if use_balance:
        moe = model.group_moe if hasattr(model, 'group_moe') else model.standard_moe
        n_opts = moe.n_options if hasattr(moe, 'n_options') else moe.router.n_options

    best_val = float("inf")
    patience = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            pred, dec = model(batch["a"].to(device), batch["op"].to(device),
                             batch["b"].to(device), batch["c"].to(device))
            loss = F.mse_loss(pred, batch["target"].to(device))
            if use_balance and dec is not None:
                loss = loss + args.balance_alpha * balance_loss(dec, n_opts)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        val_r = evaluate(model, val_loader, train_ds.denormalize, device, train_triples)
        val_loss = 1.0 - val_r.get("complement_acc", val_r["accuracy"])
        if val_loss < best_val:
            best_val = val_loss
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                break

    return evaluate(model, test_loader, train_ds.denormalize, device, train_triples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-blocks", type=int, default=2)
    parser.add_argument("--num-range", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=75)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.5)
    parser.add_argument("--balance-alpha", type=float, default=0.01)
    parser.add_argument("--output-dir", type=str, default="data/coverage_sweep")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--k-values", type=str, default="1,2,3,4",
                       help="Comma-separated k_train values (orderings per triple in training)")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("mps") if torch.backends.mps.is_available() else
        torch.device("cpu"))
    print(f"Device: {device}")

    k_values = [int(x) for x in args.k_values.split(",")]
    model_types = ["groupmoe", "standardmoe", "baseline"]
    results = {mt: {} for mt in model_types}

    for k in k_values:
        coverage = k / 6 * 100
        gen_ratio = 6 - k
        print(f"\n{'='*60}")
        print(f"k_train={k}: {coverage:.0f}% coverage, 1:{gen_ratio} generalization ratio")
        print(f"{'='*60}")

        for mt in model_types:
            torch.manual_seed(args.seed)
            print(f"  {mt}...", end=" ", flush=True)
            r = train_and_eval(mt, k, args, device)
            acc = r.get("complement_acc", 0)
            print(f"complement_acc={acc:.4f}")
            results[mt][k] = r

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print("COVERAGE SWEEP SUMMARY")
    print(f"{'='*60}")
    print(f"{'k_train':>8s} {'coverage':>9s} {'gen_ratio':>10s}", end="")
    for mt in model_types:
        print(f"  {mt:>14s}", end="")
    print()
    for k in k_values:
        line = f"{k:>8d} {k/6*100:>8.0f}% {f'1:{6-k}':>10s}"
        for mt in model_types:
            acc = results[mt].get(k, {}).get("complement_acc", 0)
            line += f"  {acc:>14.4f}"
        print(line)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"groupmoe": "blue", "standardmoe": "orange", "baseline": "green"}
    for mt in model_types:
        ks = sorted(results[mt].keys())
        accs = [results[mt][k].get("complement_acc", 0) for k in ks]
        coverages = [k / 6 * 100 for k in ks]
        ax.plot(coverages, accs, "o-", label=mt, color=colors.get(mt), linewidth=2, markersize=8)

    ax.set_xlabel("Training Coverage (% of orderings seen)", fontsize=12)
    ax.set_ylabel("Complement Transfer Accuracy", fontsize=12)
    ax.set_title("Generalization vs LUT Coverage", fontsize=13)
    ax.legend(fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(10, 75)

    # Add generalization ratio as secondary labels
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    tick_coverages = [k / 6 * 100 for k in k_values]
    ax2.set_xticks(tick_coverages)
    ax2.set_xticklabels([f"1:{6-k}" for k in k_values])
    ax2.set_xlabel("Generalization Ratio (train:test orderings)", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_dir / "coverage_curve.png", dpi=150)
    plt.close()
    print(f"\nPlot saved to {output_dir}/coverage_curve.png")


if __name__ == "__main__":
    main()
