# Group-MoE TODO

## Paper 1: Synthetic Validation ✓ COMPLETE

- [x] S_2 complement transfer (arithmetic)
- [x] S_3 complement transfer (ternary)
- [x] Three-way comparison: GroupMoE vs StandardMoE vs Baseline
- [x] Compositional generalization (transpositions → 3-cycles)
- [x] Multi-group routing (nested S_2⊂S_3, disparate Z_2/Z_3)
- [x] Transformer compatibility (drop-in FFN replacement)
- [x] General S_n representations (Young's orthogonal form, S_2–S_6)
- [x] S_n scaling experiments (identified symmetric function decomposition barrier)
- [x] Paper written, reviewed (36/40), audited (clean), published

## Paper 2: Molecular Property Prediction (NEXT)

### Phase 1: SO(3) Group Experts

- [ ] **Implement SO(3) representations**: Wigner D-matrices, spherical harmonics up to l_max=2 (total dim 9: l=0 + l=1 + l=2 = 1+3+5)
- [ ] **Adapt GroupExpert for continuous groups**: SO(3) has infinite elements — need to parameterize by rotation angle/axis rather than discrete element index
- [ ] **Design the router for continuous symmetry**: How does the router select a rotation? Options: discretize SO(3), predict Euler angles, or predict the irrep subspace weight

### Phase 2: QM9 Benchmark

- [ ] **Molecular graph construction**: atoms as nodes, bonds/distances as edges, 3D coordinates as features
- [ ] **QM9 data loader**: 134K molecules, standard 80/10/10 split, target properties (energy, HOMO-LUMO, dipole)
- [ ] **SchNet baseline**: invariant message passing (distances only)
- [ ] **SchNet + GroupMoE**: one message-passing layer replaced with GroupMoE using SO(3) expert
- [ ] **PaiNN reference**: full equivariant message passing (literature numbers or reimplementation)
- [ ] **Training and evaluation**: MAE on property predictions, router activation analysis

### Phase 3: Analysis

- [ ] **Router activation patterns**: does the router activate differently on symmetric vs asymmetric local environments?
- [ ] **Computational cost comparison**: Group-MoE vs full equivariance (parameters, FLOPs, wall time)
- [ ] **Ablation**: which properties benefit most from selective equivariance?

### Phase 4: Paper

- [ ] Write up using existing /pub pipeline
- [ ] Connect to AlphaFold and protein modeling in Discussion

## Open Research Questions

- [ ] How to handle continuous groups (SO(3)) in the discrete routing framework?
- [ ] Can the router learn to detect local point-group symmetry (C_2v, T_d, etc.) from molecular environments?
- [ ] Does selective equivariance help more on larger molecules where local symmetry varies?
- [ ] Can Group-MoE match full equivariance (PaiNN/MACE) on QM9 while being computationally cheaper?
