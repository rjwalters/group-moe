"""SO(3)-equivariant expert block.

A single expert applied at one atom: takes the atom's irrep features (l=0
scalars + l=1 vectors, optionally l=2 tensors) and returns a refined version
of the same irreps. Equivariance to SO(3) is structural — every operation is
rotation-equivariant by construction (provided by e3nn).

Block structure (per `docs/paper2_design.md`):

    input (irreps_in)
      ↓
    self-tensor-product:  irreps_in × irreps_in → intermediate
      ↓
    gated nonlinearity:   intermediate → irreps_out (= irreps_in)
      ↓
    equivariant linear mix
      ↓
    residual add to input

The "intermediate" irreps include extra scalar channels used as gates for the
non-scalar (l > 0) channels in the output. This is the standard `e3nn.nn.Gate`
pattern for equivariant nonlinearities.

Per-expert param count target (per design doc): 5–20K. The shared irreps
"16x0e + 8x1o + 4x2e" gives ~10K with this block — within range.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import nn as enn
from e3nn import o3


def _split_scalars_and_gated(irreps: o3.Irreps) -> tuple[o3.Irreps, o3.Irreps, o3.Irreps]:
    """Partition `irreps` into (scalars to activate, gates needed, gated channels).

    For e3nn's Gate: scalar (l=0, p=+1) channels get a regular activation; non-
    scalar channels each need one l=0 gate channel that gets sigmoid'd and
    multiplied in. Returns the three irreps specs in the shapes Gate expects.
    """
    irreps_scalars = o3.Irreps([(mul, ir) for mul, ir in irreps if ir.l == 0 and ir.p == 1])
    irreps_gated = o3.Irreps([(mul, ir) for mul, ir in irreps if ir.l > 0])
    # One 0e gate per non-scalar channel. Parity 0e × any → preserves parity of any.
    irreps_gates = o3.Irreps([(mul, "0e") for mul, _ in irreps_gated])
    return irreps_scalars, irreps_gates, irreps_gated


class SO3Expert(nn.Module):
    """SO(3)-equivariant residual block in a fixed irrep basis.

    Parameters:
        irreps: e3nn.o3.Irreps — the working representation. Same shape for
            input and output (residual block).

    The block applies, in order:
      1. Self-tensor-product (in × in → intermediate-with-gates)
      2. Gated nonlinearity (silu on scalars, sigmoid-gated SO(3) channels)
      3. Equivariant linear projecting back to `irreps`
      4. Residual addition with input
    """

    def __init__(self, irreps: o3.Irreps):
        super().__init__()
        self.irreps = irreps

        irreps_scalars, irreps_gates, irreps_gated = _split_scalars_and_gated(irreps)
        # Intermediate irreps the TP must produce: scalars (for direct activation)
        # + gates (one 0e per gated channel) + gated channels themselves.
        irreps_intermediate = (irreps_scalars + irreps_gates + irreps_gated).simplify()

        self.tp = o3.FullyConnectedTensorProduct(
            irreps_in1=irreps,
            irreps_in2=irreps,
            irreps_out=irreps_intermediate,
            internal_weights=True,
            shared_weights=True,
        )
        self.gate = enn.Gate(
            irreps_scalars=irreps_scalars,
            act_scalars=[F.silu] * len(irreps_scalars),
            irreps_gates=irreps_gates,
            act_gates=[torch.sigmoid] * len(irreps_gates),
            irreps_gated=irreps_gated,
        )
        # Gate output irreps = irreps_scalars + irreps_gated, which is `irreps`
        # up to ordering. Use a Linear to (a) restore canonical ordering and
        # (b) provide a learnable mixing within each l-channel.
        self.linear = o3.Linear(irreps_in=self.gate.irreps_out, irreps_out=irreps)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Apply the equivariant block.

        Args:
            features: (n_atoms, irreps.dim).

        Returns:
            (n_atoms, irreps.dim) — same shape, equivariantly updated.
        """
        h = self.tp(features, features)
        h = self.gate(h)
        h = self.linear(h)
        return features + h

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
