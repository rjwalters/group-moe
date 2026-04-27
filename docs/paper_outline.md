# Group-MoE: Learned Dispatch to Algebraic Fixed-Function Units

## Paper Outline

### Abstract

Neural networks that encounter symmetry in data either enforce it rigidly in every layer (geometric deep learning) or ignore it and hope the right behavior emerges (standard architectures). We propose a third path: Group-MoE, a mixture-of-experts architecture where each expert implements a finite group representation in the irreducible representation (irrep) basis, and a learned router detects which symmetry applies. This gives models algebraic fixed-function units — analogous to dedicated hardware accelerators — that the network learns to invoke selectively. We introduce the *complement split*, a controlled evaluation methodology that isolates symmetry transfer from memorization, and validate on synthetic tasks demonstrating: (1) complement transfer from seen to unseen orderings, (2) zero-shot compositional generalization from generator elements to composed elements, (3) that fixed irrep matrices outperform matched learned transforms, and (4) that the router learns to discriminate between non-nested groups. We connect these findings to the broader question of when algebraic structure in neural networks provides an advantage over lookup-table memorization.

### 1. Introduction

**The symmetry gap.** Our prior work ("Ordering Is Not Invariant") showed that LLMs achieve *functional* equivariance — correct outputs under input permutation — without developing *structural* equivariance — group representations in their latent space. They learn multiplication tables by rote, not algebra. This works when the table is small enough to memorize, but fails when combinatorial growth outpaces model capacity.

**Two existing approaches, both limited.**
- *Geometric deep learning* (equivariant architectures): bakes symmetry into every layer. Correct by construction but rigid — assumes the symmetry always applies, can't mix symmetric and non-symmetric computation.
- *Standard architectures*: no symmetry structure. The model can learn equivariant behavior but represents it as a lookup table, not as algebraic structure.

**Our proposal: symmetry as a menu option.** Group-MoE provides group representations as expert modules that the model can choose to invoke via a learned router. The analogy to hardware design is precise: just as processors evolved from general-purpose CPUs to include fixed-function units (FFT accelerators, matrix multiply engines), neural architectures can include algebraic fixed-function units that are cheaper and more composable than general-purpose layers, but only activated when the workload demands them.

**Contributions:**
1. The Group-MoE architecture: group experts in the irrep basis + learned symmetry router
2. The complement split: a controlled methodology for measuring symmetry transfer
3. Experimental validation: complement transfer, compositional generalization, and group discrimination
4. Analysis of when algebraic structure helps vs when lookup-table memorization suffices

### 2. Architecture

#### 2.1 Group Expert
The core building block. For a group G with irreps of total dimension k:

```
x ∈ R^d → P (d→k) → R(g) (k×k, block-diagonal, fixed) → P† (k→d) → x' ∈ R^d
```

- P and P† are learned projections into/out of the "symmetry-active" subspace
- R(g) is the irrep matrix for group element g — fixed, not learned
- Block-diagonal structure gives O(k) parameters vs O(k²) for a dense matrix
- Composition is exact by construction: R(g₁)R(g₂) = R(g₁g₂)

For S_3: k=4 (trivial + sign + standard 2D irreps), vs d=128 → 32× compression.

#### 2.2 Symmetry Router
Lightweight MLP that maps activations to a distribution over [pass-through, group₁_element₀, group₁_element₁, ...]. Hard routing (argmax) with soft confidence blending. Pass-through is the default — symmetry is applied only when the router detects it.

#### 2.3 Integration
The GroupMoELayer is a residual module: output = x + confidence × (expert(x) − x). It handles both (batch, d) and (batch, seq_len, d) inputs, making it a drop-in replacement for any FFN block. A Switch Transformer-style load-balancing loss prevents expert collapse.

### 3. The Complement Split

Our key methodological contribution. Standard train/test splits don't isolate symmetry transfer — the model might generalize for reasons unrelated to group structure (e.g., embedding interpolation).

**The complement split** ensures every test example requires *symmetry-specific* generalization:
- For S_2: if (a,+,b) is in training, only (b,+,a) is in test. Transfer requires knowing addition commutes.
- For S_3: if one ordering of {a,b,c} is in training, the other 5 are in test. Transfer requires knowing the function is permutation-invariant.
- For composition: identity + transpositions train, 3-cycles test. Transfer requires knowing how permutations compose.

In each case, a model with the correct group structure generalizes by construction; a model without it must independently learn each case.

### 4. Experiments

Each experiment is designed to answer one question. We present them in pedagogical order.

#### 4.1 Does complement transfer work? (S_2, Arithmetic)

**Setup.** (a, op, b) → result, op ∈ {+, −}. Complement split on addition pairs. Opaque embeddings, num_range=20, d_model=128.

**Result.** GroupMoE 48.4% vs Baseline 31.6% on reversed pairs — 53% relative improvement. Consistent across 3 seeds. Router discriminates: 95% S_2 routing for addition, 35% for subtraction. Per-pair analysis shows GroupMoE exclusively solves 2.5× more pairs.

**Balance loss ablation.** α=0.01 optimal. α=0 works but weaker (+30% instead of +53%). α=0.1 hurts.

*Teaches: the group expert provides genuine symmetry transfer that the baseline cannot match.*

#### 4.2 Is it the group structure or just the routing? (S_3, Three-Way Comparison)

**Setup.** (a, b, c, op) → result, op ∈ {sum, nonsym}. Three matched models:
- GroupMoE: irrep matrix R(g) (fixed)
- StandardMoE: learned k×k matrix W (same dimensions, same router, same residual blending)
- Baseline: no expert

**Result.**

| Model | Complement (1→5) | Composition (4→2) |
|-------|------------------|-------------------|
| GroupMoE (irrep R(g)) | **92.1%** | **99.0%** |
| StandardMoE (learned W) | 88.2% | 98.1% |
| Baseline | 86.1% | 96.2% |

Group structure contributes +3.9pp (complement) to +0.9pp (composition) beyond what routing architecture alone provides. The harder the generalization, the more the algebraic structure matters.

*Teaches: the advantage comes from the group structure specifically, not just from having a routing architecture.*

#### 4.3 Does irrep composition generalize? (Transpositions → 3-Cycles)

**Setup.** Train on identity + 3 transposition orderings per triple (generators of S_3). Test only on 3-cycle orderings (compositions of transpositions, never seen in training).

**Result.** GroupMoE achieves 98.5–99.0% on 3-cycles. The irrep basis provides zero-shot composition: R((012)) = R((01)) @ R((12)) is pre-defined, so the expert works for composed elements without training on them.

*Teaches: the key theoretical payoff of group representations — composition is exact by construction.*

#### 4.4 Can the router discriminate between groups?

**Nested groups (S_2 + S_3).** The router routes ~70% of all operations to S_3 regardless of operation type. Because S_2 ⊂ S_3, every S_2 transformation is also an S_3 transformation — the router is being *rational*, not failing.

**Non-nested groups (Z_2 + Z_3).** The router correctly dispatches Z_2 operations to the Z_2 expert at 76%. With non-overlapping capabilities, the router learns which ASIC to invoke.

*Teaches: the router discovers group-theoretic relationships. Subgroup containment makes discrimination unnecessary; genuine algebraic differences enable it.*

#### 4.5 Does it work in a transformer?

**Setup.** 4-layer transformer encoder, each of (a, op, b, c) as a separate token. GroupMoE replaces one FFN block.

**Result.** All models reach 100%. Self-attention handles permutation mixing natively for short sequences — it provides functional equivariance without group structure.

*Teaches: the group expert's value is marginal when the backbone already handles permutation mixing. Its advantage should manifest on tasks where attention alone cannot resolve the symmetry — longer sequences, larger groups, combinatorial state spaces.*

### 5. When Does Algebraic Structure Beat Memorization?

This is the central analytical question. Our experiments reveal a consistent pattern:

**Algebraic structure helps when:**
1. The task has genuine symmetry that the group captures
2. The complement split (or composition split) isolates transfer from memorization
3. The model's backbone doesn't already provide the needed equivariance (MLP vs attention)
4. The effective lookup table has gaps that the group structure can fill

**Memorization wins when:**
1. The function is simple enough that embeddings decompose it (sum = per-number values)
2. The model has enough capacity relative to the table size
3. Attention provides permutation mixing for free
4. Enough orderings are observed that interpolation suffices

**The LUT-vs-algebra analogy.** Models learn multiplication tables by rote before understanding algebra — and for small tables, rote memorization is strictly easier. The algebraic structure pays off when the table is too large to memorize: S_n grows as n!, while the irrep basis remains structured and compressible. Our synthetic experiments operate in the small-table regime where memorization is competitive. The compelling case for Group-MoE is the large-table regime — large groups, long sequences, combinatorial state spaces — where the factorial growth of the lookup table defeats memorization.

### 6. Related Work

- **Geometric deep learning.** Cohen & Welling (group equivariant CNNs), Bronstein et al. (geometric DL blueprint), Weiler & Cesa (steerable CNNs). Group-MoE shares the irrep decomposition machinery but makes it optional via routing, rather than mandatory in every layer.
- **Mixture of experts.** Shazeer et al. (Switch Transformer), Fedus et al., Lepikhin et al. (GShard). Experts are generic MLPs. Group-MoE gives them algebraic structure and exact composition.
- **Equivariance in transformers.** Fuchs et al. (SE(3)-Transformers), Liao & Smidt (equiformer). Domain-specific (molecules, point clouds). Group-MoE is domain-agnostic.
- **Learned symmetry.** Dehmamy et al. (automatic symmetry discovery), Zhou et al. (meta-learning symmetries), Benton et al. (learning invariances). Discover symmetries post-hoc or learn soft equivariance. Group-MoE provides exact group structure as architectural options with learned dispatch.
- **"Ordering Is Not Invariant"** (our prior work). Showed LLMs achieve functional equivariance without structural equivariance. Group-MoE is the architectural response — providing structural equivariance as an option.

### 7. Discussion and Future Work

**The ASIC analogy.** Group experts are cognitive arithmetic units — domain-specific algebraic accelerators hardwired in the irrep basis, dispatched by a learned router. The analogy to hardware evolution is not just metaphorical: the irrep basis provides the same kind of structural compression that fixed-function hardware provides over general-purpose computation.

**Scaling to combinatorial regimes.** Our synthetic experiments validate the mechanism but operate where memorization is competitive. The key prediction: for S_n with large n, the n! growth of the lookup table will defeat memorization while the irrep basis remains tractable. Testing this requires implementing larger permutation groups or finding natural tasks with large effective symmetry groups.

**Language modeling.** Natural language contains implicit symmetry: entity permutations ("Alice and Bob" vs "Bob and Alice"), fact reorderings, syntactic transforms. These create effective symmetry groups whose "multiplication tables" are too large for memorization. Inserting Group-MoE into language models and testing with entity permutation probes is the natural next step.

**Symmetry discovery.** Currently the groups are specified a priori. A compelling extension: learn which groups are useful from data, by treating the irrep structure as a differentiable hyperparameter or using a library of candidate groups with learned selection.

**Approximate symmetry.** Real data has approximate, not exact, symmetry. Soft irrep decomposition with learned tolerance could extend Group-MoE to noisy settings.

### 8. Conclusion

Group-MoE demonstrates that neural networks can learn to dispatch to algebraic fixed-function units. The complement split methodology cleanly isolates symmetry transfer from memorization. The three-way comparison shows the advantage comes from group structure specifically (+3.9pp), not just routing architecture (+2.1pp). Compositional generalization confirms the theoretical payoff: irrep composition enables zero-shot transfer. And the router learns group-theoretic relationships — subgroup containment, non-nested discrimination — without explicit supervision.

These are proof-of-concept results on synthetic tasks where memorization is competitive. The architecture's value will compound in regimes where the effective symmetry group is too large to memorize — exactly the combinatorial territory where current models struggle. Group-MoE provides the tools; finding the right large-scale application is future work.

---

## Figures

1. **Architecture diagram** — Input → Layers → Router → Group Expert / Pass-through → Layers
2. **S_2 complement transfer** — per-pair heatmaps (GroupMoE vs baseline error, advantage, routing)
3. **Three-way comparison table** — the central result (complement + composition × 3 models)
4. **Compositional generalization** — training on transpositions, testing on 3-cycles
5. **Router discrimination** — nested (S_2⊂S_3) vs non-nested (Z_2, Z_3) routing tables
6. **Transformer drop-in** — convergence curves showing compatibility
7. **When does it help?** — summary diagram of conditions for algebraic advantage

## Appendices

- A: Group representation details (irrep matrices, multiplication tables, composition verification)
- B: Hyperparameter sensitivity (balance loss α, d_model, num_range)
- C: Scaling analysis (num_range sweep, coverage sweep, discussion of LUT limits)
