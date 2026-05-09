# Paper 2: Selective SO(3) Equivariance via Group-MoE

This doc locks in the architecture for Paper 2 before any code is written. Companion to `molecular_proposal.md` (which is the high-level pitch); this is the engineering spec.

## Open question from Paper 1 → answered here

How does Group-MoE handle a continuous group like SO(3), when the existing framework routes over discrete group elements?

**Answer (option 3 from our conversation):** the router does *not* select a rotation. The router selects a **symmetry-type label**. The corresponding expert is a complete SO(3)-equivariant block whose internal operations handle continuous rotations natively (via tensor-product layers in the irrep basis).

The MoE structure is preserved: discrete categorical routing decision per token (per atom), with one of K experts (or pass-through) applied. What changes is what the experts *are* — they are equivariant sub-networks rather than `R(g)` lookup tables.

## Why this is the right choice

We considered three options:

| Option | Router job | Expert job | Verdict |
|---|---|---|---|
| 1. Discretize SO(3) to a finite subgroup (e.g. icosahedral I_h, order 60) | Classify over 60 elements | Apply `R(g)` for that element | **Rejected** — molecular rotations are isotropic; there's no preferred axis in the data, so quantizing the rotation throws away the continuous structure for no benefit |
| 2. Router predicts rotation parameters (axis-angle / quaternion) | Regress 3 continuous values | Apply Wigner D-matrix at predicted angle | **Rejected** — conflates routing (discrimination) with transformation (equivariance); the "mixture" framing weakens because there's effectively one expert with a continuous parameter |
| 3. Router emits a symmetry-type label; expert is an SO(3)-equivariant block | Classify over K + 1 categories (K symmetry types + pass-through) | Apply equivariant ops internally; the rotation is implicit in the irrep basis | **Selected** — preserves the discrete-MoE structure, matches the molecular reality (the meaningful question is *what type of local environment*, not *which rotation applies*) |

The defining claim of Group-MoE is "the model decides **when and which** symmetry to apply, not how every layer computes." Option 3 is the only one consistent with that framing in the continuous-group case.

## Architecture

### Symmetry-type labels

The router emits one of these labels per atom:

| Label | Local environment example | Irrep specialization (initial guess) |
|---|---|---|
| `tetrahedral` | sp³ carbon, tetrahedral metal complex | low-l tensor structure (l ≤ 1 sufficient) |
| `octahedral` | octahedral metal complex, sp³d² | higher-l (l ≤ 2 needed for cubic features) |
| `planar` | sp² carbon, aromatic ring atom | l = 0 + l = 1 (in-plane), l = 2 suppressed |
| `asymmetric` | chiral center, broken-symmetry environment | full l ≤ 2 with no truncation |
| `pass-through` | atom whose contribution doesn't benefit from equivariance | identity (existing module's pass-through option) |

These labels are **inductive priors on the expert architectures**, not ground-truth supervision. The router learns to dispatch from the data; the labels just constrain the experts so each one specializes to a different irrep regime.

We will start with **K = 3 experts + pass-through** (drop "asymmetric" — let pass-through handle it) and revisit if routing collapses.

### Per-atom routing

Each atom in the molecule independently gets its own routing decision based on its local activation. This is the analogue of per-token routing in language MoE.

The router input is the atom's hidden state after a few SchNet message-passing layers (so it sees its local environment). The router output is a categorical distribution over `{expert_1, ..., expert_K, pass-through}`.

### SO(3)-equivariant experts

Each expert is a small block built from `e3nn` operations:

```
input atomic features (l=0 scalar + l=1 vector)
  → tensor-product layer (irreps_in × irreps_in → irreps_out, l ≤ l_max)
  → gated nonlinearity (scalars get GELU, higher-l get scalar-times-vector gating)
  → linear in irrep basis
  → output atomic features (same irreps as input)
```

Per expert: `l_max ∈ {1, 2}`, hidden irrep multiplicity ~16. Roughly 5–20K params per expert.

Equivariance comes "for free" from `e3nn` — every operation is rotation-equivariant by construction. The router gating is a scalar (l=0), so the gated output remains equivariant.

### Integration with SchNet

```
atom embeddings (Z → 64-dim scalar features)
  ↓
SchNet message passing × 3 (invariant — distances only)
  ↓
GroupMoE layer  ← the experiment
  ↓
SchNet message passing × 3
  ↓
sum-pool over atoms
  ↓
energy prediction
```

The "no equivariance" arm omits the GroupMoE layer (or replaces it with an identity). The "rigid equivariance" arm is ViSNet (Wang et al. 2024), an equivariant vector-scalar message-passing model that's the modern PaiNN successor and is shipped in PyG. (PyG does not ship PaiNN itself; ViSNet is the closest in-tree analogue and outperforms PaiNN on QM9 in the original paper.)

## Code structure (additions only — no edits to existing modules)

```
src/groups/continuous.py        new — wraps e3nn.o3.Irreps for the SO(3) "group"
src/modules/so3_expert.py       new — replaces GroupExpert for continuous experts
src/modules/molecular_router.py new — emits K+1 categorical decision (no element_idx)
src/modules/molecular_moe.py    new — composes the above; drop-in for PyG models
src/data/qm9.py                 new — loads QM9, returns PyG Data objects
src/models/schnet.py            new — SchNet baseline (or wraps torch_geometric.nn.SchNet)
src/models/schnet_groupmoe.py   new — SchNet with one layer replaced
scripts/train_qm9.py            new — training driver
```

The existing `GroupMoELayer` stays as-is; this is a parallel module for the continuous-group case rather than a generalization. We may unify later if a clean abstraction emerges.

## Experimental plan

### Headline result

QM9 internal energy U0, MAE in meV, on the canonical 110k/10k/13k split.

| Model | Equivariance | Router? | Expected MAE (meV) |
|---|---|---|---|
| SchNet (baseline) | none | n/a | ~14 (literature); 16.7 (v5 actual) |
| SchNet + GroupMoE | selective per-atom | yes | **target: between SchNet and ViSNet** |
| ViSNet (reference) | full E(3) (lmax=1) | n/a | ~3.3 (literature) |

The Group-MoE arm doesn't need to *beat* ViSNet. It needs to (a) clearly beat SchNet on the same compute, and (b) show the router has learned interpretable per-atom symmetry detection.

### Auxiliary measurements

1. **Router activation by element:** does the router fire `tetrahedral` more on sp³ carbons than on sp² carbons? This is the interpretability win.
2. **Compute cost:** params, FLOPs, wall time vs ViSNet. Selective equivariance should be cheaper than always-on.
3. **Ablation:** routing on/off, K = 1 vs K = 3, l_max = 1 vs 2.

### Failure modes worth naming up front

- **Router collapse:** all atoms routed to one expert (or all to pass-through). Mitigation: load-balancing loss, as in standard MoE.
- **Per-atom routing too fine-grained:** noise dominates. Mitigation: average router logits over neighbors before argmax, or operate on graph-level pooled features.
- **No advantage over SchNet:** QM9 molecules are small (≤9 heavy atoms) so local symmetry rarely varies enough. Mitigation: move to MD17 (forces — equivariance more meaningful) or larger systems.

## Out of scope for Paper 2

- MD17 (forces) — Phase 2 in the proposal; defer until QM9 result is in
- Protein-scale (AlphaFold connection) — Phase 3; defer to a follow-up
- Comparison to MACE, NequIP — interesting but not needed to make the selective-equivariance point (ViSNet *is* the rigid-equivariance baseline; see Phase 1.5)

## Decision log

- **2026-05-02:** Locked in option 3 (categorical router + equivariant experts). Reason: only option consistent with the "model chooses *when* to apply symmetry" framing.
- **2026-05-02:** K = 3 experts + pass-through (tetrahedral, octahedral, planar; "asymmetric" handled by pass-through).
- **2026-05-02:** Use `e3nn` for irrep ops rather than reimplementing Wigner D / Clebsch-Gordan from scratch. Saves weeks; standard in the field.
- **2026-05-02:** SchNet as the host model (not PaiNN) so the equivariance comes *only* from the inserted GroupMoE block. Cleanly isolates the contribution.
- **2026-05-06:** ViSNet replaces PaiNN as the "rigid equivariance" reference baseline. Reason: PyG ships ViSNet but not PaiNN, and ViSNet is the modern equivariant successor (outperforms PaiNN on QM9 in the ViSNet paper). Same architectural category, no new dependencies.
