# S_n Scaling Experiment: Results and Analysis

## Hypothesis

As the group order grows (S_3 → S_4 → S_5), GroupMoE's advantage over StandardMoE should increase because the irrep basis remains structured while the learned transforms must approximate an increasingly large representation.

## Results

### Complement accuracy (mean over 3 seeds, num_range=8, d_model=128)

| Group | Order | Irrep dim | GroupMoE | StandardMoE | Baseline |
|-------|-------|-----------|---------|-------------|----------|
| S_3   | 6     | 4         | **82.7%** | 76.9%     | 80.8%    |
| S_4   | 24    | 10        | **91.2%** | 90.1%     | 90.7%    |
| S_5   | 120   | 26        | 100%    | 100%        | 100%     |

### Parameter counts

| Group | GroupMoE | StandardMoE | Ratio |
|-------|---------|-------------|-------|
| S_3   | 85K     | 85K         | 1.0x  |
| S_4   | 90K     | 93K         | 1.0x  |
| S_5   | 98K     | 179K        | 1.8x  |

## Why S_5 Hits 100%

The accuracy *increases* with n because:

1. **More training data per embedding.** With num_range=8 and n=5, there are C(12,5)=792 multisets in training. Each of the 8 numbers appears in hundreds of training examples, giving the embeddings abundant context to learn number values.

2. **Sum decomposes per-element.** The function sum(x_1,...,x_n) = x_1 + ... + x_n requires learning one value per number and one aggregation operation. The effective table size is O(num_range), not O(n!). Scaling n doesn't grow the table — it provides more data.

3. **The complement split gets easier.** At S_5, training has 792 symmetric examples, each covering one ordering out of up to 120. But since sum is invariant, the model only needs to learn what each number "means" — not how orderings relate. More multisets = more contexts per number = better embeddings.

## The Fundamental Problem

**Sum is the wrong function for a scaling study.** It has the same failure mode at S_5 that we identified in Appendix C of the paper: the function decomposes into per-element contributions, so the effective table is O(n), not O(n!).

For the scaling crossover to appear, we need a function where the output genuinely depends on the **ordering structure** in a way that creates a combinatorial lookup table.

## What Would Work

The function must satisfy:
1. **S_n-invariant** (same output under permutation) — so the complement split tests genuine symmetry transfer
2. **Not decomposable** into per-element contributions — so the model can't shortcut via embeddings
3. **Output depends on pairwise or higher-order interactions** — so the effective table grows combinatorially

### Candidate functions that DON'T work

| Function | Problem |
|----------|---------|
| sum(x_i) | Decomposes per-element: O(n) |
| max(x_i) | Decomposes per-element: O(n) |
| product(x_i) | Decomposes per-element: O(n) |
| e_2 = Σ x_i x_j | Pairwise but still O(n²) not O(n!). Also nonlinear — hard to learn from embeddings. |

### Candidate functions that MIGHT work

1. **Rank-based functions.** f(x_1,...,x_n) = index of median, or number of inversions in the sorted order. These depend on the relative ordering of values, creating genuine combinatorial structure. But they're invariant under permutation of positions (since they depend on values, not positions), which is exactly what we want.

2. **Permutation composition tasks.** Input: a sequence of transpositions. Output: the composed permutation's cycle type, or the result of applying the composed permutation to a fixed point. This directly tests R(g1)R(g2) = R(g1*g2). The effective table IS the group multiplication table (n! × n!).

3. **Graph invariant functions.** Interpret (x_1,...,x_n) as edge weights of a complete graph. Compute a graph invariant (e.g., minimum spanning tree weight). This is S_n-invariant on vertices but depends on pairwise interactions, giving O(n²) effective table size.

4. **Sorting network outputs.** f(x_1,...,x_n) = k-th element of sorted sequence. This is S_n-invariant (the sorted sequence doesn't depend on input order) but requires comparing all pairs — the effective table grows with the number of distinct orderings of values.

### Most promising: Permutation composition

This is the most natural candidate because:
- The output directly uses the group multiplication table
- The table IS n! × n! (genuinely combinatorial)
- The irrep basis computes it exactly by construction
- No learned transform can guarantee R(g1)R(g2) = R(g1*g2)
- It tests the unique theoretical advantage of the irrep basis

**Design:** Input two permutations σ, τ ∈ S_n (encoded as sequences of transpositions or as permutation tuples). Output: some property of σ∘τ (e.g., its sign, its fixed-point count, or the full composed permutation).

This requires a fundamentally different dataset design — the inputs are group elements, not numbers. But it directly tests whether the group expert can compose in the irrep basis while the baseline must memorize the multiplication table.
