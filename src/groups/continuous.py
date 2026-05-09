"""SO(3) symmetry-type configs for Paper 2.

In Paper 1 the router selected a discrete group element and the expert applied
R(g) in a learned subspace. For SO(3) (continuous, isotropic rotations) that
formulation breaks down — there's no preferred axis, no element index. Instead
the router selects a *symmetry-type label* (tetrahedral / octahedral / planar /
pass-through) and the corresponding expert is a small SO(3)-equivariant block
whose internal ops handle rotations natively in the irrep basis (via e3nn).

This module is the configuration layer: each SymmetryType carries the e3nn
`Irreps` spec that defines the expert's working representation. The actual
equivariant ops live in `src/modules/so3_expert.py`.

The discrete `GroupRepresentation` API (element_idx, multiplication_table) is
intentionally not implemented — the continuous case has no analogue. See
`docs/paper2_design.md` decision log for the framing.
"""

from __future__ import annotations

from dataclasses import dataclass

from e3nn import o3


@dataclass(frozen=True)
class SymmetryType:
    """Inductive prior on an SO(3)-equivariant expert.

    `irreps` is the expert's working representation: l=0 channels carry scalar
    features (invariants), l=1 channels carry vectors (transform under rotation),
    l=2 channels carry rank-2 tensors (e.g. cubic-symmetry features).

    The expert's input and output irreps are both `irreps` (same), so it can
    be inserted as a residual block.
    """
    name: str
    irreps: o3.Irreps
    description: str

    def __repr__(self) -> str:
        return f"SymmetryType({self.name}, irreps={self.irreps})"


# --- Preset symmetry types matching the design doc ------------------------
#
# Multiplicities (16, 8, 4) chosen to match SchNet v5 baseline's hidden=256
# scalar capacity at the l=0 channel (16 * 16 = 256-equivalent expressivity)
# while keeping per-expert param counts small (~5-20K, per the design doc).

TETRAHEDRAL = SymmetryType(
    name="tetrahedral",
    irreps=o3.Irreps("16x0e + 8x1o"),
    description="sp3 environments — full 3D symmetry, rotation-vector basis sufficient (l ≤ 1).",
)

OCTAHEDRAL = SymmetryType(
    name="octahedral",
    irreps=o3.Irreps("16x0e + 8x1o + 4x2e"),
    description="octahedral / cubic environments — d-orbital-like features need rank-2 tensors (l ≤ 2).",
)

PLANAR = SymmetryType(
    name="planar",
    irreps=o3.Irreps("16x0e + 8x1o"),
    description="sp2 / aromatic environments — in-plane vector basis sufficient; rank-2 tensors suppressed.",
)


# Default expert set: K=3 + pass-through (handled by molecular_moe, not here).
# Order is meaningful: routing label index → preset.
DEFAULT_SYMMETRY_TYPES: list[SymmetryType] = [TETRAHEDRAL, OCTAHEDRAL, PLANAR]


def shared_irreps(types: list[SymmetryType]) -> o3.Irreps:
    """Smallest irreps spec containing every irrep in the given symmetry types.

    The molecular MoE layer needs a common input/output irreps so all experts
    can be applied to the same atom features. We compute the union of all
    types' irreps with the maximum multiplicity per (l, parity) channel.

    For DEFAULT_SYMMETRY_TYPES this gives "16x0e + 8x1o + 4x2e" — the union
    of the three presets.
    """
    # Build a dict {(l, parity): max_mul} across all types
    mul_by_lp: dict[tuple[int, int], int] = {}
    for t in types:
        for mul, ir in t.irreps:
            key = (ir.l, ir.p)
            mul_by_lp[key] = max(mul_by_lp.get(key, 0), mul)
    # Sort by l then parity to get a canonical ordering
    pieces = []
    for (l, p), mul in sorted(mul_by_lp.items()):
        ir = o3.Irrep(l, p)
        pieces.append(f"{mul}x{ir}")
    return o3.Irreps(" + ".join(pieces))
