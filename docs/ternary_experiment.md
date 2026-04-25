# Ternary Experiment: S_3 Complement Transfer

## Hypothesis

If the S_3 group expert correctly implements the permutation representation, then a model trained on one ordering of (a, b, c) should generalize to all 5 remaining orderings for S_3-invariant functions. This extends the S_2 arithmetic result (1→1 transfer) to S_3 (1→5 transfer).

## Setup

- **Task**: predict `f(a, b, c)` for two operations:
  - `symmetric(a,b,c) = a + b + c` — fully S_3-invariant
  - `nonsym(a,b,c) = 2a - b + c` — position-dependent, no symmetry
- **Architecture**: identical to arithmetic experiment but with 4 inputs (a, op, b, c) and S_3 expert (7 router options: 1 pass-through + 6 S_3 elements)
- **Complement split**: for each unordered multiset {a,b,c}, exactly 1 ordering in training, rest (up to 5) in test. Non-symmetric op split randomly.

## Results

### Function choice matters

**e_2(a,b,c) = ab + ac + bc** (first attempt): both models plateau at ~11% complement accuracy. The nonlinear function is too hard to learn from opaque embeddings at this scale — the model spends all its capacity learning number products rather than exploiting symmetry.

**sum(a,b,c) = a + b + c** (final choice): models learn well, enabling a clean symmetry test.

### S_3 complement transfer (num_range=15, d_model=128)

| Seed | GroupMoE +comp | Baseline +comp | Gap |
|------|---------------|----------------|-----|
| 42   | **91.8%**     | 86.4%          | +5.4pp |
| 123  | **91.9%**     | 90.6%          | +1.3pp |

GroupMoE consistently outperforms baseline on complement transfer, though the gap is smaller than the S_2 experiment (~5pp vs ~16pp).

### Router behavior

**Seed 42**: S_3 rate 86% for symmetric, 88% for non-symmetric — weak discrimination, expert used broadly as extra compute.

**Seed 123**: S_3 rate 99.9% for symmetric, 80% for non-symmetric — strong discrimination, the router correctly identifies which operation benefits from the group structure.

### Why the gap is smaller than S_2

1. **1→5 transfer is harder than 1→1**: S_2 has one non-trivial element (swap). S_3 has five. The correct element depends on which specific permutation maps the test ordering to the training ordering — information the router doesn't have.
2. **Router complexity**: 7 routing options vs 3 for S_2. More options → harder optimization.
3. **Baseline is stronger**: with num_range=15, there are enough training examples per number that the baseline can partially generalize through embedding interpolation alone.

### num_range=10 fails

At num_range=10 (C(10,3)=120 distinct triples, ~220 symmetric training examples), both models plateau at ~69% with no GroupMoE advantage. The issue is insufficient data for embedding learning, not the architecture.

## Conclusions

1. **S_3 complement transfer works** — GroupMoE provides a genuine advantage on ternary S_3-invariant functions, extending the S_2 arithmetic result.
2. **The effect is weaker than S_2** — 1→5 transfer is inherently harder, and the router has more options to navigate.
3. **Scale matters** — num_range=15 succeeds where num_range=10 fails. The architecture needs enough data to learn embeddings before the group structure can help.
4. **Function choice matters** — nonlinear functions (e_2) are too hard for the model to learn at this scale. Linear functions (sum) work.
5. **Router discrimination is seed-dependent** — some seeds produce clean operation discrimination (99% vs 80%), others don't (86% vs 88%). The balance loss helps but doesn't guarantee discrimination.

## Comparison with S_2 Arithmetic

| Metric | S_2 (arithmetic) | S_3 (ternary) |
|--------|-----------------|---------------|
| Group order | 2 | 6 |
| Complement ratio | 1→1 | 1→5 |
| Best complement gap | ~16pp | ~5pp |
| Router discrimination | 95%/35% | 99%/80% |
| Min viable num_range | 20 | 15 |
