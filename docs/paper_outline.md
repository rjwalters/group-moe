# Group-MoE: Learned Dispatch to Algebraic Fixed-Function Units

## Paper Outline

### Abstract

Neural networks either bake symmetry into every layer (geometric deep learning — rigid) or ignore it entirely (standard architectures — wasteful). We propose Group-MoE, a mixture-of-experts architecture where each expert implements a group representation in the irreducible representation basis, and a learned router detects which symmetry applies to the current input. This gives models algebraic fixed-function units — analogous to dedicated hardware accelerators — that the network learns to invoke selectively. We validate on synthetic tasks showing complement transfer (generalization from one ordering to unseen permutations), router discrimination between non-nested groups, and parameter efficiency from irrep compression.

### 1. Introduction

- **Problem**: LLMs learn functional equivariance without structural equivariance ("Ordering Is Not Invariant"). Geometric deep learning enforces structure everywhere. Neither is ideal.
- **Insight**: Group representations as optional expert modules — learned ASICs for algebraic operations. The router is the learned dispatch controller.
- **Analogy**: just as fixed-function hardware (FFT units, MAC arrays) provides efficiency for specific operations, irrep-basis group experts provide compressed, exact algebraic transformations that general-purpose layers must approximate expensively.
- **Contribution**: architecture + synthetic validation showing the router learns when and which symmetry to apply.

### 2. Architecture

- **Group Expert module**: project to irrep subspace (d → k), apply block-diagonal R(g) (k × k), project back (k → d). Parameter-efficient by construction.
- **Symmetry Router**: lightweight MLP over activations, outputs distribution over [pass-through, group1_e0, group1_e1, ..., group2_e0, ...]. Hard routing with soft confidence blending.
- **Residual integration**: output = x + confidence × (expert(x) − x). Pass-through is the default.
- **Load-balancing loss**: Switch Transformer-style auxiliary loss to prevent expert collapse.
- **Implemented groups**: Z_2, Z_3, S_2, S_3 with verified composition tables.

### 3. Experimental Setup: Complement Transfer

Key methodological contribution: the **complement split**.

- For each unordered set/pair, exactly one ordering goes to training; remaining orderings go to test.
- Every test example requires transferring from a seen ordering to an unseen one.
- The group expert should help because it knows the symmetry by construction; the baseline must learn it from data.

This isolates the group structure's contribution from general memorization.

### 4. Results

#### 4.1 S_2 Complement Transfer (Arithmetic)
- **Task**: (a, op, b) → result, op ∈ {+, −}, num_range=20
- **Finding**: GroupMoE 48.4% vs baseline 31.6% on reversed addition pairs. ~50% relative improvement. Consistent across seeds.
- **Router**: 95% S_2 for addition, 35% for subtraction — clean discrimination.
- **Per-pair analysis**: GroupMoE exclusively solves 2.5× more pairs than baseline.
- **Balance loss ablation**: α=0.01 sweet spot. α=0 still works (weaker). α=0.1 hurts.

#### 4.2 S_3 Three-Way Comparison (Ternary)
- **Task**: (a, b, c, op) → result, op ∈ {sum, nonsym}, num_range=15
- **Key ablation**: three-way comparison isolates the group structure's contribution:

  | Model | Complement (1→5) | Composition (4→2) |
  |-------|------------------|-------------------|
  | GroupMoE (fixed irrep R(g)) | **92.1%** | **99.0%** |
  | StandardMoE (learned W) | 88.2% | 98.1% |
  | Baseline (no expert) | 86.1% | 96.2% |

- **Decomposition**: on complement split, group structure contributes +3.9pp, routing architecture +2.1pp. When generalization is harder, algebraic structure matters more.
- **Failure mode**: nonlinear symmetric functions (e_2) unlearnable from embeddings. Linear functions (sum) work.
- **Scale matters**: num_range=10 fails, num_range=15 succeeds.

#### 4.3 Multi-Group Routing: Nested Groups (S_2 + S_3)
- **Task**: 3 operations with S_3, S_2, and no symmetry
- **Finding**: S_3 expert dominates all routing (~70% for all ops). Router cannot discriminate because S_2 ⊂ S_3 — routing everything to S_3 is optimal.
- **Insight**: nested groups make discrimination unnecessary. The more expressive expert subsumes the simpler one.

#### 4.4 Multi-Group Routing: Disparate Groups (Z_2 + Z_3)
- **Task**: 3 operations with Z_2 symmetry (linear), Z_3 symmetry (cubic), no symmetry
- **Finding**: Router correctly dispatches Z_2 ops to Z_2 expert at 76% — first clean group-specific routing. Z_3 function unlearnable (cubic too hard), leaving Z_3 routing as future work.
- **Insight**: non-nested groups enable genuine dispatch discrimination. The router learns which ASIC to invoke when the units have non-overlapping capabilities.

#### 4.6 Transformer Integration
- **Task**: same S_3 ternary task, but with a 4-layer transformer encoder (self-attention + FFN). Each of (a, op, b, c) is a separate token. GroupMoE replaces one FFN.
- **Finding**: all three models (GroupMoE, StandardMoE, Baseline) reach 100% complement accuracy. Self-attention on 4 tokens handles permutation mixing natively.
- **What this shows**: GroupMoE is a zero-degradation drop-in FFN replacement — architectural compatibility confirmed. But for short sequences where attention can see all tokens, the group expert's marginal value is low.
- **What this means**: the MLP experiments are the correct testbed for isolating the group structure's contribution. The group expert's advantage would manifest in transformers on tasks where attention alone can't resolve the symmetry — e.g., symmetry among a subset of tokens in a longer sequence.

### 5. Analysis

- **When does Group-MoE help?** When (1) the task has genuine symmetry, (2) the model can learn the underlying function, (3) the complement split isolates transfer, (4) the group expert provides structure the baseline lacks, and (5) the architecture doesn't already handle permutation mixing (i.e., MLP over concatenated inputs, not attention over token sequences).
- **Router behavior**: learns soft preferences, not hard rules. Discriminates operations by ~10-20pp, not 100%/0%. Uses larger experts as general-purpose compute.
- **Failure modes**: nonlinear functions too hard for embeddings; nested groups make discrimination pointless; too little data starves embedding learning; attention on short sequences makes the group expert redundant.
- **Parameter efficiency**: group experts use O(d × k) parameters where k = Σ irrep dims, vs O(d²) for full matrices. For S_3: k=4 vs d=128 → 32× compression.

### 6. Related Work

- **Geometric deep learning**: Cohen & Welling (group equivariant CNNs), Bronstein et al. (geometric deep learning blueprint). Rigid — symmetry everywhere. Group-MoE makes it optional.
- **Mixture of experts**: Shazeer et al. (Switch Transformer), Fedus et al. Experts are generic MLPs. Group-MoE gives them algebraic structure.
- **Equivariance in transformers**: Equivariant attention (Fuchs et al.), SE(3)-Transformers. Domain-specific (molecules, point clouds). Group-MoE is domain-agnostic.
- **Learned symmetry detection**: Dehmamy et al. (automatic symmetry discovery), Zhou et al. (meta-learning symmetries). Discover symmetries post-hoc. Group-MoE provides them as architectural options.
- **"Ordering Is Not Invariant"** (our prior work): showed LLMs lack structural equivariance. Group-MoE is the architectural response.

### 7. Discussion and Future Work

- **Cognitive arithmetic units**: Group experts as learned fixed-function hardware — domain-specific algebraic accelerators with learned dispatch. Extends the ASIC analogy: just as hardware evolves from general-purpose CPUs to specialized accelerators, neural architectures can evolve from generic layers to algebraic expert modules.
- **Compositional generalization at scale**: test the zero-shot composition property on larger groups and real tasks.
- **Language modeling**: insert Group-MoE into a transformer. Does the router activate on entity permutations in natural language?
- **Symmetry discovery**: can the architecture discover which groups are useful, rather than being told? Auto-construction of expert modules from data.
- **Approximate symmetry**: real data has approximate, not exact, symmetry. Soft irrep decomposition, continuous group parameters.
- **Scale**: does the compression advantage compound at scale? Larger models, larger groups, more experts.

#### 4.5 Compositional Generalization (Transpositions → 3-Cycles)
- **Task**: same ternary task, but train on identity + transposition orderings (4 per triple), test ONLY on 3-cycle orderings (2 per triple, which are compositions of transpositions)
- **Finding**: GroupMoE 98.5-99.0% on 3-cycles vs baseline 97.3-97.6%. Near-perfect generalization to composed elements never seen during training.
- **Router**: 100% S_3 for symmetric ops at num_range=10 — perfect group identification.
- **Insight**: the irrep basis provides zero-shot composition. R((012)) = R((01)) @ R((12)) is pre-defined, so the expert works for 3-cycles without ever training on them. The projection P, learned from transposition examples, generalizes because the irrep subspace is the same for all elements.
- **Tradeoff**: GroupMoE sacrifices ~5pp on non-symmetric ops to achieve near-perfect compositional generalization on symmetric ops.

### 8. Conclusion

Group-MoE demonstrates that neural networks can learn to dispatch to algebraic fixed-function units. The complement transfer results show genuine structural advantage from group representations. The compositional generalization result confirms the key theoretical prediction: irrep composition enables zero-shot transfer to composed group elements. The router discrimination results show that non-nested groups enable learned specialization. The architecture provides a path from "hoping symmetry emerges" to "providing symmetry as a menu option."

---

## Status of Evidence

| Claim | Status | Evidence |
|-------|--------|----------|
| S_2 complement transfer | **Strong** | 48% vs 32%, 3 seeds, balance ablation, per-pair analysis |
| S_3 complement transfer | **Strong** | 92.1% vs 88.2% vs 86.1% (3-way), group structure = +3.9pp |
| Compositional generalization | **Strong** | 99.0% vs 98.1% vs 96.2% (3-way), zero-shot composition |
| Group > Standard MoE | **Strong** | Group structure contributes +0.9pp (composition) to +3.9pp (complement) beyond routing architecture |
| Router discriminates operations | **Moderate** | 95%/35% for S_2; 100%/74% for composition split |
| Router discriminates non-nested groups | **Preliminary** | 76% Z_2 dispatch; Z_3 task unlearnable |
| Nested groups → no discrimination | **Strong** | S_2 ⊂ S_3 proven theoretically + empirically |
| Parameter efficiency | **Architectural** | k=4 for S_3 vs d=128 → 32× compression (by construction) |
| Transformer compatibility | **Strong** | Zero-degradation drop-in FFN replacement; all models reach 100% |
| Language modeling benefit | **Not yet tested** | — |

## What's Needed Before Submission

1. ~~**Compositional generalization experiment**~~ ✓ Done — near-perfect zero-shot composition
2. ~~**Comparison to standard MoE**~~ ✓ Done — group structure contributes +3.9pp beyond routing on complement split
3. ~~**Transformer integration**~~ ✓ Done — zero-degradation drop-in, confirms architectural compatibility
4. **A learnable Z_3 task** — to complete the disparate-groups story (nice-to-have, not blocking)
5. **Longer-sequence transformer task** — where attention can't trivially solve the symmetry (future work)
