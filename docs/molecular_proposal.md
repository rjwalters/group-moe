# Group-MoE for Molecular Property Prediction: Proposal

## Motivation

Current equivariant molecular models (SchNet, PaiNN, MACE, ViSNet) enforce SE(3) equivariance in every layer. This is correct but rigid — every computation is equally equivariant, regardless of local molecular symmetry.

Real molecular environments have *variable* local symmetry:
- High symmetry: aromatic rings (C_6v), tetrahedral carbons (T_d), octahedral metal complexes (O_h)
- Low symmetry: chiral centers (C_1), mutations in proteins, crystal defects
- Approximate symmetry: helical repeats, pseudo-symmetric binding sites

Group-MoE can selectively apply equivariance per atom or per edge, using the learned router to detect local symmetry and dispatch to the appropriate group expert. This is the "optional symmetry" thesis applied to a real domain.

## Connection to AlphaFold

AlphaFold3 and subsequent models use:
- SE(3)-equivariant features for structure representation
- Invariant pair features for distance/angle prediction
- Attention mechanisms that are permutation-equivariant on the residue sequence

Group-MoE could enhance these at the residue level:
- **Router detects local symmetry** of each residue's environment
- **Group expert applies** the appropriate point-group representation
- **Pass-through** for residues in asymmetric environments

This would give AlphaFold-style models the ability to exploit local symmetry without enforcing it globally.

## Proposed Experiment

### Phase 1: QM9 Benchmark (feasibility)

**Task:** Predict molecular properties (energy, HOMO-LUMO gap, dipole moment) from 3D atomic coordinates.

**Architecture:**
```
Atom features → Message Passing Layers → [GroupMoE Layer] → Message Passing → Readout
```

Three models:
1. **SchNet baseline** — invariant message passing (distances only)
2. **SchNet + GroupMoE** — one layer replaced with GroupMoE using SO(3) spherical harmonics irreps
3. **ViSNet reference** — full equivariant message passing (modern PaiNN successor; PyG-native)

**Group experts for molecules:**
- SO(3) spherical harmonics: l=0 (invariant), l=1 (vector), l=2 (tensor) — total dim = 1+3+5 = 9
- Or discrete molecular point groups: C_2v, C_3v, T_d (if we detect molecular symmetry)

**Dataset:** QM9 (134K molecules, up to 9 heavy atoms). Standard 80/10/10 split.

**What to measure:**
- MAE on property predictions (compare to SchNet and ViSNet)
- Router activation patterns: does the router learn to activate on symmetric local environments?
- Computational cost: Group-MoE should be faster than full equivariance (fewer parameters, selective application)

### Phase 2: MD17 (dynamics)

**Task:** Predict forces on atoms in molecular dynamics trajectories.

Forces are equivariant (they transform as vectors under rotation). This is where the Group-MoE expert should provide the strongest advantage — the expert applies the rotation representation exactly, while SchNet can only learn rotation-invariant features.

### Phase 3: Protein-Scale (AlphaFold connection)

**Task:** Predict per-residue properties (B-factors, contact maps) from protein structure.

This connects to AlphaFold by showing that Group-MoE can handle the variable symmetry of protein environments at scale.

## Technical Requirements

### New group representations needed
- **SO(3) truncated to l_max**: spherical harmonics up to order l_max. These are continuous groups, unlike the finite groups we've implemented. Would need:
  - Wigner D-matrices for rotation representations
  - Clebsch-Gordan coefficients for tensor products
  - Or: use the existing finite-group framework with discrete rotational symmetry (e.g., icosahedral group I_h, order 60)

### Dataset infrastructure
- Molecular graph construction from coordinates
- Edge features (distances, optional angles)
- Atom features (element type, charge)
- QM9 loader (available as PyG or DGL dataset)

### Dependencies
- `torch_geometric` or `torch_scatter` for graph operations
- QM9 dataset (auto-downloadable via PyG)

## Comparison to Existing Work

| Approach | Equivariance | When Applied | Our Advantage |
|----------|-------------|--------------|---------------|
| SchNet | Invariant only | Always | We add selective equivariance |
| PaiNN / ViSNet | Full E(3) equivariant | Always, everywhere | We apply only where detected |
| MACE | Higher-order equivariant | Always, everywhere | We adapt per-atom |
| **Group-MoE** | **Selective, per-atom** | **Router-detected** | **Optional symmetry** |

## Estimated Effort

- Phase 1 (QM9): 2-3 weeks (new code for molecular graphs, SO(3) representations, training pipeline)
- Phase 2 (MD17): 1 week additional (similar pipeline, different task)
- Phase 3 (proteins): major effort (new data loading, larger models, longer training)

## Risk Assessment

**High risk:** The advantage of selective equivariance may not materialize on QM9 — the molecules are small enough that full equivariance (ViSNet) isn't wasteful. The benefit should be clearer on larger systems (proteins, materials).

**Medium risk:** Implementing SO(3) representations properly is non-trivial. Could use e3nn library as a reference or dependency.

**Low risk:** The architecture (routing + irrep experts) is proven from our synthetic experiments. The question is whether it helps on real molecular data.
