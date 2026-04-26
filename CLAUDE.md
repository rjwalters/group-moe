# Group-MoE

## Project Overview

Group-structured mixture of experts: an architecture where each expert module implements a specific group representation in the irreducible representation (irrep) basis, and a learned router detects which symmetry applies to the current input.

This is a follow-up to the "Ordering Is Not Invariant" paper (see `../latent-space-symmetries/`), which showed that LLMs learn functional equivariance (correct outputs under permutation) without structural equivariance (group representations in latent space). Group-MoE aims to close this gap by providing group structure as an architectural option the model can choose to use.

## Key Idea

Instead of building equivariance into every layer (rigid, the geometric deep learning approach) or hoping it emerges (it doesn't, per our findings), Group-MoE lets the model **choose when and which symmetry to apply** via a mixture-of-experts architecture where each expert is a group representation module.

## Architecture

```
Input → Standard Layers → Symmetry Router → Group Expert (or pass-through) → Standard Layers
```

- **Router**: lightweight classifier detecting which group action (if any) applies
- **Group Experts**: implement R(g) in irrep basis — block-diagonal, parameter-efficient
- **Pass-through**: for inputs where no symmetry applies

## Why This Could Work

- **100x parameter reduction** for symmetry-bearing transformations (irrep basis vs full matrix)
- **Compositional generalization by construction** — group representations compose correctly
- **Self-supervised symmetry detection** — the router learns to use group experts because they're cheaper
- **Selective application** — symmetry used when data has it, bypassed when it doesn't

## Research Phases

1. ~~**Implement core modules**~~: Group representations (S_2, S_3, Z_2, Z_3), irrep decomposition, router ✓
2. ~~**Synthetic validation**~~: Complement transfer (S_2, S_3), compositional generalization, multi-group routing ✓
3. **Comparison**: Group-MoE vs standard MoE vs equivariant architectures on compositional tasks
4. **Scale**: Test on language modeling with entity permutation and fact reordering

## Conventions

- Use `uv` for environment management
- Core library in `src/`, experiments in `scripts/`, tests in `tests/`
- Save checkpoints and data under `data/` (gitignored)
