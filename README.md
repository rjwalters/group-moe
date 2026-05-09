# Group-MoE

**Group-structured Mixture of Experts**: an architecture where each expert implements a group representation in the irreducible representation (irrep) basis, and a learned router detects which symmetry applies to the current input.

This is a follow-up to the ["Ordering Is Not Invariant"](https://github.com/rwalters/latent-space-symmetries) paper, which showed that LLMs learn functional equivariance (correct outputs under permutation) without structural equivariance (group representations in latent space). Group-MoE closes this gap by making group structure an architectural option the model can choose to use.

## Motivation

Existing approaches to symmetry in neural networks sit at two extremes:

| Approach | Problem |
|---|---|
| **Geometric deep learning** (equivariant layers everywhere) | Rigid. Assumes the symmetry always applies. |
| **Standard architectures** (hope equivariance emerges) | It doesn't. Models learn the right input-output map but not the right internal structure. |

Group-MoE takes a third path: give the model **group representations as a menu of experts**, and let it learn when to apply them.

```
Input --> Standard Layers --> Symmetry Router --> Group Expert (or pass-through) --> Standard Layers
```

- **Router** -- lightweight MLP that detects which group action (if any) applies
- **Group Experts** -- apply R(g) in the irrep basis: block-diagonal, parameter-efficient
- **Pass-through** -- for inputs where no symmetry applies

## Why this works

- **Parameter efficiency**: irrep matrices are block-diagonal, giving ~100x compression vs full d x d matrices. For example, S_3 uses a 4 x 4 block-diagonal matrix instead of a full d x d matrix per group element.
- **Exact composition**: group representations compose correctly by construction -- R(g1) @ R(g2) = R(g1 * g2). No approximation, no learning required.
- **Selective symmetry**: the router learns to activate group experts only when the data has that symmetry. No rigid constraint on the full model.

## Implemented groups

| Group | Order | Irreps | Total dim |
|---|---|---|---|
| Z_2 | 2 | trivial (1D) + sign (1D) | 2 |
| Z_3 | 3 | trivial (1D) + standard (2D) | 3 |
| S_2 | 2 | trivial (1D) + sign (1D) | 2 |
| S_3 | 6 | trivial (1D) + sign (1D) + standard (2D) | 4 |

## Quick start

```bash
# Clone and set up
git clone https://github.com/rwalters/group-moe.git
cd group-moe
uv sync

# Run tests
uv run pytest
```

## Usage

```python
import torch
from src.groups.representations import S3Representation, Z2Representation
from src.modules.group_moe import GroupMoELayer

# Create a Group-MoE layer with S_3 and Z_2 experts
layer = GroupMoELayer(
    d_model=256,
    groups=[S3Representation(), Z2Representation()],
)

# Forward pass -- router automatically selects the right expert
x = torch.randn(32, 256)
output, decision = layer(x)

# Inspect routing decisions
stats = layer.router.routing_stats(decision)
print(stats)
# {'pass_through_rate': 0.75, 'mean_confidence': 0.82, 'S_3_rate': 0.15, 'Z_2_rate': 0.10}

# Check parameter efficiency
print(layer.param_summary())
```

The layer also handles sequence inputs (batch, seq_len, d_model) and can be inserted into a transformer as a drop-in layer.

## Project structure

```
src/
  groups/
    representations.py   # Group representations in irrep basis (Z_2, Z_3, S_2, S_3)
  modules/
    expert.py            # GroupExpert: project -> R(g) -> inject
    router.py            # SymmetryRouter: detects which group/element applies
    group_moe.py         # GroupMoELayer: combines router + experts
  data/
    arithmetic.py        # S_2 complement transfer dataset
    ternary.py           # S_3 complement + composition datasets
    multigroup.py        # S_2 + S_3 multi-group dataset
    disparate.py         # Z_2 + Z_3 non-nested group dataset
  models/
    arithmetic.py        # S_2 arithmetic models
    ternary.py           # S_3 ternary models
    multigroup.py        # Multi-group (S_2 + S_3) models
    disparate.py         # Disparate-group (Z_2 + Z_3) models
scripts/
  train_arithmetic.py    # S_2 complement transfer experiment
  train_ternary.py       # S_3 complement + composition experiment
  train_multigroup.py    # Multi-group routing experiment
  train_disparate.py     # Disparate-group routing experiment
  analyze_complement.py  # Per-pair analysis of S_2 results
tests/
  test_groups.py         # Verify irrep matrices, composition tables
  test_modules.py        # Expert, router, and full layer tests
  test_arithmetic.py     # Arithmetic dataset + model tests
  test_ternary.py        # Ternary dataset + model tests
  test_multigroup.py     # Multi-group dataset + model tests
  test_disparate.py      # Disparate-group dataset + model tests
docs/
  paper_outline.md       # Paper outline with evidence inventory
  arithmetic_experiment.md
  ternary_experiment.md
```

## Experimental results

Each experiment isolates a single question about the architecture:

| Experiment | Question | Key result |
|---|---|---|
| S_2 arithmetic | Does complement transfer work? | **48% vs 32%** on reversed addition pairs |
| S_3 ternary | Does it scale to larger groups? | **92% vs 86%** on permuted triples |
| Composition split | Does irrep composition generalize? | **98.5%** on unseen 3-cycles (zero-shot) |
| S_2 + S_3 multi-group | Can the router discriminate nested groups? | No — S_2 ⊂ S_3, so routing to S_3 is optimal |
| Z_2 + Z_3 disparate | Can it discriminate non-nested groups? | **76%** correct Z_2 dispatch |

See `docs/paper_outline.md` for the full paper structure and evidence inventory.

## Research roadmap

### Paper 1: Synthetic Validation ✓

1. ~~Complement transfer~~ (S_2: +16pp, S_3: routing architecture +3.2pp)
2. ~~Three-way comparison~~ (GroupMoE vs StandardMoE vs Baseline, 5 seeds)
3. ~~Compositional generalization~~ (98.7% zero-shot on 3-cycles)
4. ~~Multi-group routing~~ (nested and non-nested discrimination)
5. ~~Transformer compatibility~~ (zero-degradation drop-in)
6. ~~General S_n representations~~ (Young's orthogonal form, verified through S_6)

Published: `paper/group-moe.5/` and [rjwalters.info/research/2026-group-moe](https://rjwalters.info/research/2026-group-moe)

### Paper 2: Molecular Property Prediction (in progress)

1. **SO(3) group experts** -- spherical harmonics irreps for continuous rotation symmetry, applied via tensor-product layers in the irrep basis (e3nn)
2. **QM9 benchmark** -- selective equivariance vs rigid equivariance (ViSNet) vs none (SchNet)
3. **Per-atom routing** -- categorical router emits a symmetry-type label (tetrahedral / octahedral / planar / pass-through) per atom
4. **Path to proteins** -- variable local symmetry in AlphaFold-style models

See `docs/paper2_design.md` for the architecture spec and `docs/molecular_proposal.md` for the full plan.
Cloud training (Lambda Labs A100/A10) is set up via `scripts/lambda_train.sh` — see `docs/cloud_training.md`.

## How the expert works

The `GroupExpert` applies symmetry in a learned subspace:

```
x in R^d  -->  P (project)  -->  z in R^k  -->  R(g)  -->  z' in R^k  -->  P† (inject)  -->  x' in R^d
```

Where k is the sum of irrep dimensions (e.g., 4 for S_3) and P is a learned linear projection. This means the group action is applied in a tiny subspace, while the rest of the activation space passes through unchanged.

## License

MIT
