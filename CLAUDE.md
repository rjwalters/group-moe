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

## Research Program

### Paper 1: Synthetic Validation (COMPLETE)

Demonstrated complement transfer, compositional generalization, and router discrimination on synthetic tasks. Published at `paper/group-moe.5/` and on rjwalters.info. Key finding: the routing architecture robustly helps (+3.2pp), while fixed irreps provide theoretical composition guarantees but don't consistently beat learned transforms at toy scale.

### Paper 2: Molecular Property Prediction (NEXT)

Apply Group-MoE to QM9 molecular energy prediction with SO(3) group experts. Test selective equivariance (router-detected, per-atom) against rigid equivariance (PaiNN, MACE) and no equivariance (SchNet). See `docs/molecular_proposal.md` for the full plan. Path toward AlphaFold-style protein modeling.

### Key Lessons Learned

- **Symmetric functions on numbers decompose** via sorted multisets — can't produce O(n!) scaling challenge. Need physics problems with genuine non-decomposable symmetry.
- **Attention provides functional equivariance natively** on short sequences — group experts add value only when the backbone can't handle permutation mixing.
- **The LUT-vs-algebra crossover** requires the effective lookup table to grow faster than model capacity. Real molecular systems (variable local symmetry, continuous rotations) are the right domain.

## Implemented Groups

| Group | Type | Module | Status |
|-------|------|--------|--------|
| Z_2, Z_3 | Finite cyclic | `representations.py` | Complete |
| S_2, S_3 | Finite symmetric (hand-coded) | `representations.py` | Complete |
| S_n (general) | Finite symmetric (Young's orthogonal) | `symmetric.py` | Complete |
| SO(3) | Continuous rotation | `continuous.py` | TODO (Paper 2) |

## Conventions

- Use `uv` for environment management
- Core library in `src/`, experiments in `scripts/`, tests in `tests/`
- Save checkpoints and data under `data/` (gitignored)
- Papers in `paper/` with immutable version history (see `.claude/skills/pub/SKILL.md`)
- Publication pipeline: `/pub-draft` → `/pub-review` → `/pub-revise` → `/pub-audit` → `/pub-website`
