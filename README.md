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
    representations.py   # Group representations in irrep basis (Z_2, S_2, S_3)
  modules/
    expert.py            # GroupExpert: project -> R(g) -> inject
    router.py            # SymmetryRouter: detects which group/element applies
    group_moe.py         # GroupMoELayer: combines router + experts
tests/
  test_groups.py         # Verify irrep matrices, composition tables
  test_modules.py        # Expert, router, and full layer tests
scripts/                 # Experiment scripts (forthcoming)
```

## Research roadmap

1. **Synthetic validation** -- train on tasks with known symmetry, verify the router learns to detect it
2. **Multiple groups** -- D_4, cyclic groups, multi-group routing and interaction
3. **Language modeling** -- integrate into a small transformer, test on entity permutation tasks
4. **Scale** -- parameter efficiency and convergence benchmarks at 1B+ parameters

## How the expert works

The `GroupExpert` applies symmetry in a learned subspace:

```
x in R^d  -->  P (project)  -->  z in R^k  -->  R(g)  -->  z' in R^k  -->  P† (inject)  -->  x' in R^d
```

Where k is the sum of irrep dimensions (e.g., 4 for S_3) and P is a learned linear projection. This means the group action is applied in a tiny subspace, while the rest of the activation space passes through unchanged.

## License

MIT
