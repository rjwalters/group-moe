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

### Phase 0: Design and infrastructure

- [x] **Design doc** locking in the SO(3) router/expert architecture (`docs/paper2_design.md`)
- [x] **QM9 data loader** with canonical 110k/10k/10.8k split, atomref subtraction (`src/data/qm9.py`)
- [x] **Dependencies**: torch_geometric, e3nn, ase, torch-cluster, torch-scatter

### Phase 1: SchNet baseline (in progress)

- [ ] **SchNet baseline training run**: invariant message passing, U0 prediction, target ~14 meV MAE — running in background as the reference number for the Group-MoE arm

### Phase 2: SO(3) Group Experts

- [ ] **`src/groups/continuous.py`**: thin wrapper around e3nn.o3.Irreps so SO(3) fits the existing GroupRepresentation interface
- [ ] **`src/modules/so3_expert.py`**: SO(3)-equivariant block (tensor-product layer, gated nonlinearity, irreps up to l_max=2)
- [ ] **`src/modules/molecular_router.py`**: per-atom categorical router emitting K+1 symmetry-type labels (no element_idx)
- [ ] **`src/modules/molecular_moe.py`**: composes router + experts; drop-in for PyG message-passing models

### Phase 3: SchNet + GroupMoE

- [ ] **`src/models/schnet_groupmoe.py`**: SchNet with one interaction block replaced by the molecular MoE layer
- [ ] **Training run**: same hyperparameters as the baseline, on the same split
- [ ] **PaiNN reference**: pull literature MAE or run a quick reimplementation for the upper-bound comparison

### Phase 4: Analysis

- [ ] **Router activation patterns**: does the router activate differently on sp³ vs sp² carbons, on aromatic ring atoms, on hydrogens? (the interpretability win)
- [ ] **Compute cost**: params, FLOPs, wall time vs PaiNN — selective equivariance should be cheaper than always-on
- [ ] **Ablation**: routing on/off, K = 1 vs K = 3, l_max = 1 vs 2

### Phase 5: Paper

- [ ] Write up using existing /pub pipeline
- [ ] Connect to AlphaFold and protein modeling in Discussion

### Phase 3: Analysis

- [ ] **Router activation patterns**: does the router activate differently on symmetric vs asymmetric local environments?
- [ ] **Computational cost comparison**: Group-MoE vs full equivariance (parameters, FLOPs, wall time)
- [ ] **Ablation**: which properties benefit most from selective equivariance?

### Phase 4: Paper

- [ ] Write up using existing /pub pipeline
- [ ] Connect to AlphaFold and protein modeling in Discussion

## Open Research Questions

- [x] How to handle continuous groups (SO(3)) in the discrete routing framework? → categorical router over symmetry-type labels + SO(3)-equivariant experts (see `docs/paper2_design.md`)
- [ ] Can the router learn to detect local point-group symmetry (C_2v, T_d, etc.) from molecular environments?
- [ ] Does selective equivariance help more on larger molecules where local symmetry varies?
- [ ] Can Group-MoE match full equivariance (PaiNN/MACE) on QM9 while being computationally cheaper?
