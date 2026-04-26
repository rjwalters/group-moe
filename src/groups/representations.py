"""Group representations in the irreducible representation basis.

Each GroupRepresentation provides:
- The list of irreps with their dimensions
- For each group element, the block-diagonal matrix in the irrep basis
- Composition: R(g1) @ R(g2) == R(g1 * g2) by construction

The key efficiency gain: instead of d x d matrices, we store only the
irrep blocks (total dimension k << d). The embedding into the full
d-dimensional space is handled by the GroupExpert module.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import numpy as np


@dataclass
class Irrep:
    """An irreducible representation."""
    name: str
    dim: int


class GroupRepresentation(ABC):
    """Base class for group representations in the irrep basis."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the group."""

    @property
    @abstractmethod
    def order(self) -> int:
        """Number of elements in the group."""

    @property
    @abstractmethod
    def irreps(self) -> list[Irrep]:
        """List of irreducible representations used."""

    @property
    def total_dim(self) -> int:
        """Total dimension of the representation (sum of irrep dims)."""
        return sum(ir.dim for ir in self.irreps)

    @abstractmethod
    def element_names(self) -> list[str]:
        """Names for each group element (for routing)."""

    @abstractmethod
    def irrep_matrix(self, element_idx: int) -> torch.Tensor:
        """Block-diagonal irrep matrix for the given group element.

        Returns a (total_dim, total_dim) matrix that is block-diagonal
        with one block per irrep.
        """

    def all_matrices(self) -> torch.Tensor:
        """All irrep matrices stacked: (order, total_dim, total_dim)."""
        return torch.stack([self.irrep_matrix(i) for i in range(self.order)])

    def verify_composition(self, atol: float = 1e-6) -> bool:
        """Verify that R(g1) @ R(g2) == R(g1*g2) for all pairs."""
        matrices = self.all_matrices()
        table = self.multiplication_table()
        for i in range(self.order):
            for j in range(self.order):
                product = matrices[i] @ matrices[j]
                expected = matrices[table[i, j]]
                if not torch.allclose(product, expected, atol=atol):
                    return False
        return True

    @abstractmethod
    def multiplication_table(self) -> torch.LongTensor:
        """(order, order) table where table[i,j] = index of element_i * element_j."""


class Z2Representation(GroupRepresentation):
    """Z_2 = {e, g} with g^2 = e.

    Irreps: trivial (1D) + sign (1D) = 2D total.
    """

    @property
    def name(self) -> str:
        return "Z_2"

    @property
    def order(self) -> int:
        return 2

    @property
    def irreps(self) -> list[Irrep]:
        return [Irrep("trivial", 1), Irrep("sign", 1)]

    def element_names(self) -> list[str]:
        return ["e", "g"]

    def irrep_matrix(self, element_idx: int) -> torch.Tensor:
        if element_idx == 0:  # identity
            return torch.eye(2)
        else:  # g: trivial=+1, sign=-1
            return torch.diag(torch.tensor([1.0, -1.0]))

    def multiplication_table(self) -> torch.LongTensor:
        return torch.tensor([[0, 1], [1, 0]], dtype=torch.long)


class Z3Representation(GroupRepresentation):
    """Z_3 = {e, r, r^2} with r^3 = e. Cyclic group of order 3.

    Irreps over R: trivial (1D) + standard (2D, rotation by 2π/3) = 3D total.
    """

    @property
    def name(self) -> str:
        return "Z_3"

    @property
    def order(self) -> int:
        return 3

    @property
    def irreps(self) -> list[Irrep]:
        return [Irrep("trivial", 1), Irrep("standard", 2)]

    def element_names(self) -> list[str]:
        return ["e", "r", "r^2"]

    def irrep_matrix(self, element_idx: int) -> torch.Tensor:
        theta = 2 * np.pi * element_idx / 3
        c, s = np.cos(theta), np.sin(theta)
        block = torch.zeros(3, 3)
        block[0, 0] = 1.0  # trivial
        block[1, 1] = c    # standard 2D rotation
        block[1, 2] = -s
        block[2, 1] = s
        block[2, 2] = c
        return block

    def multiplication_table(self) -> torch.LongTensor:
        # r^i * r^j = r^((i+j) mod 3)
        return torch.tensor([
            [0, 1, 2],
            [1, 2, 0],
            [2, 0, 1],
        ], dtype=torch.long)


class S2Representation(GroupRepresentation):
    """S_2 = {e, (01)} — same as Z_2 but named as a permutation group.

    Irreps: trivial (1D) + sign (1D) = 2D total.
    """

    @property
    def name(self) -> str:
        return "S_2"

    @property
    def order(self) -> int:
        return 2

    @property
    def irreps(self) -> list[Irrep]:
        return [Irrep("trivial", 1), Irrep("sign", 1)]

    def element_names(self) -> list[str]:
        return ["e", "(01)"]

    def irrep_matrix(self, element_idx: int) -> torch.Tensor:
        if element_idx == 0:
            return torch.eye(2)
        else:
            return torch.diag(torch.tensor([1.0, -1.0]))

    def multiplication_table(self) -> torch.LongTensor:
        return torch.tensor([[0, 1], [1, 0]], dtype=torch.long)


class S3Representation(GroupRepresentation):
    """S_3 with all three irreps: trivial (1D) + sign (1D) + standard (2D) = 4D total.

    Elements indexed as:
      0: e (identity)
      1: (01) — transposition, odd
      2: (12) — transposition, odd
      3: (02) — transposition, odd
      4: (012) — 3-cycle, even
      5: (021) — 3-cycle, even
    """

    @property
    def name(self) -> str:
        return "S_3"

    @property
    def order(self) -> int:
        return 6

    @property
    def irreps(self) -> list[Irrep]:
        return [Irrep("trivial", 1), Irrep("sign", 1), Irrep("standard", 2)]

    def element_names(self) -> list[str]:
        return ["e", "(01)", "(12)", "(02)", "(012)", "(021)"]

    def irrep_matrix(self, element_idx: int) -> torch.Tensor:
        # Trivial: all -> 1
        # Sign: even -> +1, odd -> -1
        # Standard 2D: s=(01) -> reflection, r=(012) -> 120° rotation

        c = np.cos(2 * np.pi / 3)
        s = np.sin(2 * np.pi / 3)

        # Standard 2D irrep matrices
        std = {
            0: np.eye(2),                          # e
            1: np.array([[1, 0], [0, -1]]),         # (01) = s
            2: np.array([[-0.5, -s], [-s, 0.5]]),   # (12) = r*s (verify: rs)
            3: np.array([[-0.5, s], [s, 0.5]]),     # (02) = s*r (verify: sr)
            4: np.array([[c, -s], [s, c]]),          # (012) = r
            5: np.array([[c, s], [-s, c]]),          # (021) = r^2
        }

        parity = [0, 1, 1, 1, 0, 0]  # 0=even, 1=odd
        sign_val = 1.0 if parity[element_idx] == 0 else -1.0

        # Build block-diagonal: [trivial(1) | sign(1) | standard(2)]
        block = torch.zeros(4, 4)
        block[0, 0] = 1.0                    # trivial
        block[1, 1] = sign_val               # sign
        std_mat = torch.tensor(std[element_idx], dtype=torch.float32)
        block[2:4, 2:4] = std_mat            # standard 2D

        return block

    def multiplication_table(self) -> torch.LongTensor:
        # S_3 multiplication table (row * col)
        # Elements: 0=e, 1=(01), 2=(12), 3=(02), 4=(012), 5=(021)
        return torch.tensor([
            [0, 1, 2, 3, 4, 5],  # e *
            [1, 0, 4, 5, 2, 3],  # (01) *
            [2, 5, 0, 4, 3, 1],  # (12) *
            [3, 4, 5, 0, 1, 2],  # (02) *
            [4, 3, 1, 2, 5, 0],  # (012) *
            [5, 2, 3, 1, 0, 4],  # (021) *
        ], dtype=torch.long)
