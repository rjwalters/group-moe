"""General S_n representations via Young's orthogonal form.

Computes irreducible representations for any symmetric group S_n
programmatically from the partition structure. No hand-coding needed.

Algorithm:
1. Enumerate partitions of n → one irrep per partition
2. For each partition, enumerate standard Young tableaux (SYT)
3. Build adjacent-transposition matrices using axial distances
4. Compose for arbitrary permutations via bubble-sort decomposition
5. Assemble block-diagonal irrep matrix from all irreps
"""

from __future__ import annotations

from itertools import permutations as iter_permutations
from math import factorial, sqrt

import numpy as np
import torch

from .representations import GroupRepresentation, Irrep


# ---------------------------------------------------------------------------
# Partition enumeration
# ---------------------------------------------------------------------------

def partitions(n: int) -> list[tuple[int, ...]]:
    """All partitions of n in reverse lexicographic order."""
    if n == 0:
        return [()]
    result: list[tuple[int, ...]] = []

    def _gen(remaining: int, max_part: int, current: list[int]) -> None:
        if remaining == 0:
            result.append(tuple(current))
            return
        for part in range(min(remaining, max_part), 0, -1):
            _gen(remaining - part, part, current + [part])

    _gen(n, n, [])
    return result


def hook_length_dim(partition: tuple[int, ...]) -> int:
    """Dimension of the irrep for this partition via hook length formula."""
    n = sum(partition)
    # Conjugate partition
    conj = [0] * (partition[0] if partition else 0)
    for part in partition:
        for j in range(part):
            conj[j] += 1

    # Hook lengths
    product = 1
    for i, row_len in enumerate(partition):
        for j in range(row_len):
            hook = (row_len - j) + (conj[j] - i) - 1
            product *= hook

    return factorial(n) // product


# ---------------------------------------------------------------------------
# Standard Young Tableaux
# ---------------------------------------------------------------------------

def standard_young_tableaux(shape: tuple[int, ...]) -> list[list[list[int]]]:
    """All standard Young tableaux of the given shape.

    Returns list of tableaux. Each tableau is a list of rows,
    each row a list of entries (1-indexed, strictly increasing).
    """
    n = sum(shape)
    nrows = len(shape)
    row_lengths = [0] * nrows
    tab: dict[tuple[int, int], int] = {}
    results: list[list[list[int]]] = []

    def _fill(k: int) -> None:
        if k > n:
            results.append([[tab[(r, c)] for c in range(shape[r])] for r in range(nrows)])
            return
        for r in range(nrows):
            c = row_lengths[r]
            if c >= shape[r]:
                continue
            if r > 0 and row_lengths[r - 1] <= c:
                continue
            tab[(r, c)] = k
            row_lengths[r] += 1
            _fill(k + 1)
            row_lengths[r] -= 1
            del tab[(r, c)]

    _fill(1)
    return results


# ---------------------------------------------------------------------------
# Young's Orthogonal Representation
# ---------------------------------------------------------------------------

def _position_of(tableau: list[list[int]], k: int) -> tuple[int, int]:
    """Return (row, col) of value k in the tableau (0-indexed)."""
    for r, row in enumerate(tableau):
        for c, val in enumerate(row):
            if val == k:
                return (r, c)
    raise ValueError(f"Value {k} not found in tableau")


def _content(tableau: list[list[int]], k: int) -> int:
    """Content of cell containing k: col - row (0-indexed)."""
    r, c = _position_of(tableau, k)
    return c - r


def _swap_values(tableau: list[list[int]], i: int, j: int) -> list[list[int]] | None:
    """Swap values i and j in the tableau. Return new tableau if valid SYT, else None."""
    new_tab = [list(row) for row in tableau]
    ri, ci = _position_of(tableau, i)
    rj, cj = _position_of(tableau, j)
    new_tab[ri][ci] = j
    new_tab[rj][cj] = i

    # Check SYT property: rows and columns strictly increasing
    for r, row in enumerate(new_tab):
        for c in range(len(row) - 1):
            if row[c] >= row[c + 1]:
                return None
        for c in range(len(row)):
            if r > 0 and c < len(new_tab[r - 1]) and new_tab[r - 1][c] >= new_tab[r][c]:
                return None
    return new_tab


def _tableau_key(tableau: list[list[int]]) -> tuple[tuple[int, ...], ...]:
    """Hashable key for a tableau."""
    return tuple(tuple(row) for row in tableau)


class YoungOrthogonalIrrep:
    """One irreducible representation of S_n via Young's orthogonal form."""

    def __init__(self, partition: tuple[int, ...]):
        self.partition = partition
        self.n = sum(partition)
        self.tableaux = standard_young_tableaux(partition)
        self.dim = len(self.tableaux)

        # Map tableau key → index
        self._tab_index = {_tableau_key(t): i for i, t in enumerate(self.tableaux)}

        # Precompute matrices for adjacent transpositions s_i (swaps i and i+1, 1-indexed)
        self._adj_trans_matrices: list[np.ndarray] = []
        for i in range(1, self.n):  # s_1, s_2, ..., s_{n-1}
            self._adj_trans_matrices.append(self._build_adjacent_transposition(i))

    def _build_adjacent_transposition(self, i: int) -> np.ndarray:
        """Build matrix for adjacent transposition s_i (swaps values i and i+1)."""
        d = self.dim
        M = np.zeros((d, d))

        processed = set()
        for idx, T in enumerate(self.tableaux):
            if idx in processed:
                continue

            rho = _content(T, i + 1) - _content(T, i)

            # Try swapping values i and i+1
            T_swapped = _swap_values(T, i, i + 1)
            if T_swapped is not None:
                key = _tableau_key(T_swapped)
                idx2 = self._tab_index[key]

                # 2x2 block
                s = sqrt(1.0 - 1.0 / (rho * rho))
                M[idx, idx] = 1.0 / rho
                M[idx, idx2] = s
                M[idx2, idx] = s
                M[idx2, idx2] = -1.0 / rho

                processed.add(idx)
                processed.add(idx2)
            else:
                # 1x1 block
                M[idx, idx] = 1.0 / rho
                processed.add(idx)

        return M

    def matrix_for_permutation(self, sigma: tuple[int, ...]) -> np.ndarray:
        """Compute irrep matrix for permutation sigma (0-indexed tuple).

        sigma is a tuple where sigma[i] = j means position i maps to value j.
        """
        # Decompose into adjacent transpositions via bubble sort
        # We need to express sigma as a product of adjacent transpositions s_i
        # (1-indexed: s_i swaps positions i-1 and i, i.e., values at those positions)

        # Convert to 1-indexed for the representation
        # sigma as 0-indexed: sigma[i] is the value at position i
        # We need the sequence of adjacent transpositions that sorts sigma back to identity

        arr = list(sigma)
        trans_indices: list[int] = []

        # Bubble sort: each swap (j-1, j) corresponds to s_j (1-indexed)
        for i in range(len(arr)):
            for j in range(len(arr) - 1, i, -1):
                if arr[j] < arr[j - 1]:
                    arr[j], arr[j - 1] = arr[j - 1], arr[j]
                    trans_indices.append(j)  # s_j (1-indexed: swaps values at pos j-1, j)

        # sigma = s_{t[-1]} * ... * s_{t[1]} * s_{t[0]}
        # R(sigma) = R(s_{t[-1]}) @ ... @ R(s_{t[0]})
        # But bubble sort gives: identity = s_{t[0]} * s_{t[1]} * ... * s_{t[-1]} * sigma
        # So sigma = s_{t[-1]}^{-1} * ... * s_{t[0]}^{-1} = s_{t[-1]} * ... * s_{t[0]}
        # (since transpositions are self-inverse)
        # R(sigma) = R(s_{t[-1]}) @ ... @ R(s_{t[0]})

        if not trans_indices:
            return np.eye(self.dim)

        result = self._adj_trans_matrices[trans_indices[-1] - 1].copy()
        for k in range(len(trans_indices) - 2, -1, -1):
            result = result @ self._adj_trans_matrices[trans_indices[k] - 1]

        return result


# ---------------------------------------------------------------------------
# General S_n Representation
# ---------------------------------------------------------------------------

class SnRepresentation(GroupRepresentation):
    """General S_n representation computed via Young's orthogonal form.

    Produces a block-diagonal irrep matrix for each of the n! permutations.
    One block per partition of n, in reverse lexicographic order.
    """

    def __init__(self, n: int):
        assert n >= 2, "S_n requires n >= 2"
        self._n = n
        self._partitions = partitions(n)
        self._irrep_objects = [YoungOrthogonalIrrep(p) for p in self._partitions]
        self._irrep_list = [
            Irrep(name=str(p), dim=obj.dim)
            for p, obj in zip(self._partitions, self._irrep_objects)
        ]
        self._total_dim = sum(ir.dim for ir in self._irrep_list)

        # Enumerate all permutations in canonical order
        self._elements = list(iter_permutations(range(n)))
        self._elem_to_idx = {e: i for i, e in enumerate(self._elements)}

        # Precompute all irrep matrices
        self._matrices = self._precompute_all()

    def _precompute_all(self) -> torch.Tensor:
        """Precompute block-diagonal matrices for all elements."""
        matrices = torch.zeros(self.order, self._total_dim, self._total_dim)

        for elem_idx, sigma in enumerate(self._elements):
            offset = 0
            for irrep_obj in self._irrep_objects:
                d = irrep_obj.dim
                block = irrep_obj.matrix_for_permutation(sigma)
                matrices[elem_idx, offset:offset + d, offset:offset + d] = torch.tensor(
                    block, dtype=torch.float32
                )
                offset += d

        return matrices

    @property
    def name(self) -> str:
        return f"S_{self._n}"

    @property
    def order(self) -> int:
        return factorial(self._n)

    @property
    def irreps(self) -> list[Irrep]:
        return self._irrep_list

    def element_names(self) -> list[str]:
        """Cycle notation for each permutation."""
        names = []
        for sigma in self._elements:
            if sigma == tuple(range(self._n)):
                names.append("e")
                continue
            # Compute cycle notation
            visited = [False] * self._n
            cycles = []
            for i in range(self._n):
                if visited[i] or sigma[i] == i:
                    continue
                cycle = []
                j = i
                while not visited[j]:
                    visited[j] = True
                    cycle.append(j)
                    j = sigma[j]
                if len(cycle) > 1:
                    cycles.append("(" + "".join(str(x) for x in cycle) + ")")
            names.append("".join(cycles) if cycles else "e")
        return names

    def irrep_matrix(self, element_idx: int) -> torch.Tensor:
        return self._matrices[element_idx]

    def all_matrices(self) -> torch.Tensor:
        return self._matrices

    def multiplication_table(self) -> torch.LongTensor:
        n = self._n
        order = self.order
        table = torch.zeros(order, order, dtype=torch.long)

        for i, g1 in enumerate(self._elements):
            for j, g2 in enumerate(self._elements):
                # g1 * g2: apply g2 first, then g1
                product = tuple(g1[g2[k]] for k in range(n))
                table[i, j] = self._elem_to_idx[product]

        return table
