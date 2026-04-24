# Group-MoE TODO

## Phase 1: Synthetic Validation (next)

- [ ] **Synthetic task with known symmetry**: Design a toy sequence-to-sequence task where inputs contain entity permutations with known S_3 structure. Train a small transformer with and without Group-MoE layer. Verify the router learns to activate the S_3 expert on permutation-bearing inputs.
- [ ] **Symmetry detection accuracy**: Measure router precision/recall — does it correctly identify when symmetry applies vs when it doesn't?
- [ ] **Compositional generalization test**: Train on a subset of S_3 elements, test on held-out compositions. Group-MoE should generalize; baseline should not.
- [ ] **Ablation: group expert vs generic MLP expert**: Same parameter budget, does the group structure help?

## Phase 2: Multiple Groups

- [ ] **Add more groups**: D_4 (spatial reasoning), cyclic groups C_n, product groups
- [ ] **Multi-group routing**: Verify the router can discriminate between different symmetries (e.g., S_2 entity swap vs Z_2 negation) in the same model
- [ ] **Group interaction**: What happens when an input has both S_2 and Z_2 symmetry? Sequential application? Product group expert?

## Phase 3: Language Modeling

- [ ] **Integrate with a small transformer**: Insert Group-MoE layers into a 50M-parameter transformer trained on text
- [ ] **Entity permutation probes**: Use the probe prompts from `../latent-space-symmetries/` to test whether the Group-MoE model develops structural equivariance (unlike vanilla transformers)
- [ ] **Fact-order sensitivity comparison**: Does Group-MoE handle fact reordering differently from vanilla? Does the router activate on fact permutations?

## Phase 4: Efficiency & Scale

- [ ] **Parameter efficiency benchmarks**: Compare Group-MoE vs standard MoE vs dense at matched parameter counts
- [ ] **Training efficiency**: Does Group-MoE converge faster on symmetry-bearing tasks?
- [ ] **Scale to 1B+**: Does the compression ratio advantage matter more at scale?

## Open Research Questions

- [ ] Does the router learn symmetry detection via the efficiency incentive alone, or does it need explicit supervision?
- [ ] How to handle approximate symmetry (soft irrep decomposition)?
- [ ] Can the irrep subspace projection P be shared across groups?
- [ ] Is there a way to discover new groups from data rather than specifying them a priori?
