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

### Phase 1: SchNet baseline ✓ COMPLETE

- [x] **SchNet baseline training run** (v5, 2026-05-05): 1000 epochs cosine on Lambda A10, hidden=256, 8 interactions. **Test MAE 16.7 meV** (best val 16.4 meV) — ~19% off literature ~14 meV, ~7% above the 15.5 meV target. Train ≈ val (no overfitting). Cost ~$16, 12.5h wall. Results at `data/qm9/schnet_baseline_v5/`. See `memory/project_qm9_baseline.md`.

### Phase 1.5: ViSNet reference baseline

- [ ] **ViSNet reference run**: the "rigid equivariance" arm for the three-way Paper 2 comparison. PyG does **not** ship PaiNN; it ships `torch_geometric.nn.models.ViSNet` (Wang et al. 2024), the modern equivariant successor that beats PaiNN on QM9 (~3.3 meV on U0 vs PaiNN's ~5.85 meV in the original papers). Script: `scripts/train_qm9_visnet.py`, matched recipe (hidden=256, num_layers=8, lmax=1, batch=100, AdamW lr 5e-4 → 1e-7, 1000 epochs cosine) on the 110k/10k/10.8k split. **Caveat**: ViSNet at matched config is ~8.76M params (4× SchNet's 2.2M), so expect ~25–35h on A10, ~$32–50.

### Phase 2: SO(3) Group Experts ✓ COMPLETE (2026-05-07)

- [x] **`src/groups/continuous.py`**: SymmetryType configs (frozen dataclass) + presets TETRAHEDRAL/OCTAHEDRAL/PLANAR + `shared_irreps()` union helper. The discrete `GroupRepresentation` interface deliberately not implemented — no element_idx in the continuous case.
- [x] **`src/modules/so3_expert.py`**: SO(3)-equivariant residual block (self-TP → gated nonlinearity → o3.Linear). 10–13K params per expert. Equivariance verified numerically (rel error 1e-5).
- [x] **`src/modules/molecular_router.py`**: per-atom categorical MLP, K+1 outputs. Takes only scalar (l=0) features → routing is rotation-invariant by construction. Includes `load_balancing_loss()` (Switch Transformer style).
- [x] **`src/modules/molecular_moe.py`**: dispatches per-atom to one of K experts or pass-through. Confidence-weighted blending. Returns (output, decision, lb_loss). Full-stack equivariance verified (rel error 6e-7).
- [x] **`src/modules/scalar_to_irrep.py`** (new, not in original plan): lifting layer that creates l>0 features from neighbor geometry (SchNet's scalars don't have vectors). One-block PaiNN-style: edge SH × learned scalar weights × cosine envelope, scatter-summed to atoms. ~6K params, equivariant at machine precision.

### Phase 3: SchNet + GroupMoE

- [x] **`src/models/schnet_groupmoe.py`**: subclasses `torch_geometric.nn.SchNet`. Inserts lift + MoE + scalar-projection-back at `moe_position` in the interaction loop. Zero-init reducer = near-identity at training start. Forward returns (energy, decision, lb_loss). End-to-end smoke test passes on synthetic molecules.
- [x] **Real-QM9 integration test**: 8-batch CPU run (2026-05-07). Loss descends 1227 → 204 eV; routing distribution shifts as router learns; backprop, atomref, mean/std, batched edges all wired correctly.
- [x] **Training script `scripts/train_qm9_groupmoe.py`**: clone of `train_qm9_visnet.py` with backbone swap, `--include-irrep-norms` flag, and lb_loss added to training loss.
- [x] **Headline training runs** on Lambda — **negative result confirmed**.
   - **v1** (2026-05-07): terminated at e133 / best val 657 meV / pure trajectory failure. `memory/project_groupmoe_v1.md`.
   - **v2 sweep** (2026-05-07/08): 3 parallel variants on A10. v2a (lr=1e-5) → 92 meV; v2b (lb=0, router collapsed = SchNet alone) → 29 meV; v2c (norm reducer) → 186 meV @ e419 (terminated). **In every config where MoE is active, model is worse than when MoE is bypassed.** Diagnosis: scalar bottleneck on a scalar-output task. `memory/project_groupmoe_v2_sweep.md`, `docs/paper2_routes_forward.md`.
- [x] **ViSNet reference** — v4 final best val **8.7 meV at e993**, train_mae 2.1 meV. ~36h on A100, ~$72. Output at `data/qm9/visnet_baseline_v4/`. Memory: `project_visnet_v4_baseline.md`.

### Phase 3.5: Routes to a positive result (status as of 2026-05-09)

See `docs/paper2_routes_forward.md` for full analysis.

- [x] **Route 1 (MD17 forces) — attempted on Mac, abandoned.** MPS wall-time grew 240s → 820s over 15 epochs across three configurations (h=128/L=4, MPS empty_cache fix, h=64/L=3 aggressive downscale). MPS doesn't have efficient kernels for autograd-of-autograd through e3nn. Lambda would cost ~$50 for 36h; deferred until justified.
- [x] **Route 2 (MoE-in-ViSNet) — code built, pre-flight analysis says skip.** Implementation: `src/models/visnet_groupmoe.py` and `scripts/train_qm9_visnet_groupmoe.py` (smoke-tested, equivariant). Pre-flight analysis (`scripts/analyze_visnet_v4.py`) found per-atom ‖v‖ CV=0.18 overall, 0.16 within carbons — well below the 0.5 threshold for "router has signal." ViSNet's equivariance is uniform across atoms; per-atom MoE has nothing to specialize on QM9. Skipped the $40 training run.
- [ ] **Route 3 (deferred): Discrete chemical point-group experts.** T_d, D_3h, C_3v, C_2v via Paper 1's GroupRepresentation framework. Highest infra burden, most uncertain payoff.

**Current decision: paper from existing data.** Three architecturally-distinct attempts (SchNet+MoE v1, v2 sweep with 3 variants, ViSNet+MoE pre-flight analysis) all point to the same conclusion: QM9 energy is too invariant for selective equivariance to add value on top of an already-equivariant baseline. MD17 remains the natural follow-up if we want a positive result, but requires Lambda (~$50).

### Phase 4: Analysis

- [ ] **Router activation patterns**: does the router activate differently on sp³ vs sp² carbons, on aromatic ring atoms, on hydrogens? (the interpretability win)
- [ ] **Compute cost**: params, FLOPs, wall time vs ViSNet — selective equivariance should be cheaper than always-on
- [ ] **Ablation**: routing on/off, K = 1 vs K = 3, l_max = 1 vs 2

### Phase 5: Paper

- [ ] Write up using existing /pub pipeline. Honest framing: "Selective equivariance via Group-MoE: a negative result on QM9 and what it tells us." Three architecturally-distinct attempts converge; ViSNet uniformity analysis explains why.
- [ ] Connect to AlphaFold and protein modeling in Discussion (where forces and explicit per-residue symmetry types could vindicate the approach)

### Phase 4: Paper

- [ ] Write up using existing /pub pipeline
- [ ] Connect to AlphaFold and protein modeling in Discussion

## Open Research Questions

- [x] How to handle continuous groups (SO(3)) in the discrete routing framework? → categorical router over symmetry-type labels + SO(3)-equivariant experts (see `docs/paper2_design.md`)
- [ ] Can the router learn to detect local point-group symmetry (C_2v, T_d, etc.) from molecular environments?
- [ ] Does selective equivariance help more on larger molecules where local symmetry varies?
- [ ] Can Group-MoE match full equivariance (ViSNet/MACE) on QM9 while being computationally cheaper?
