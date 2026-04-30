"""Tests for general S_n representations via Young's orthogonal form."""

import torch
import pytest
from math import factorial

from src.groups.symmetric import (
    SnRepresentation, partitions, standard_young_tableaux, hook_length_dim,
)


class TestPartitions:
    def test_partition_counts(self):
        # Number of partitions: 1, 1, 2, 3, 5, 7, 11, 15
        assert len(partitions(0)) == 1
        assert len(partitions(1)) == 1
        assert len(partitions(2)) == 2
        assert len(partitions(3)) == 3
        assert len(partitions(4)) == 5
        assert len(partitions(5)) == 7
        assert len(partitions(6)) == 11

    def test_partitions_sum(self):
        for n in range(1, 7):
            for p in partitions(n):
                assert sum(p) == n

    def test_partitions_decreasing(self):
        for n in range(1, 7):
            for p in partitions(n):
                for i in range(len(p) - 1):
                    assert p[i] >= p[i + 1]


class TestSYT:
    def test_syt_counts_match_hook_length(self):
        """Number of SYTs should match hook length formula dimension."""
        for n in range(2, 7):
            for p in partitions(n):
                syts = standard_young_tableaux(p)
                dim = hook_length_dim(p)
                assert len(syts) == dim, (
                    f"Partition {p}: {len(syts)} SYTs != {dim} from hook length"
                )

    def test_syt_entries(self):
        """Each SYT should contain exactly 1..n."""
        for p in [(3, 1), (2, 2), (3, 2), (4, 1)]:
            n = sum(p)
            for tab in standard_young_tableaux(p):
                entries = sorted(val for row in tab for val in row)
                assert entries == list(range(1, n + 1))


class TestSnRepresentation:
    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def test_order(self, n):
        g = SnRepresentation(n)
        assert g.order == factorial(n)

    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def test_total_dim(self, n):
        g = SnRepresentation(n)
        expected = {2: 2, 3: 4, 4: 10, 5: 26}
        assert g.total_dim == expected[n], f"S_{n}: total_dim={g.total_dim}, expected={expected[n]}"

    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def test_identity_is_identity(self, n):
        g = SnRepresentation(n)
        identity = tuple(range(n))
        idx = g._elem_to_idx[identity]
        R = g.irrep_matrix(idx)
        assert torch.allclose(R, torch.eye(g.total_dim), atol=1e-6), f"S_{n}: identity matrix is not I"

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_composition(self, n):
        """R(g1) @ R(g2) == R(g1*g2) for all pairs."""
        g = SnRepresentation(n)
        assert g.verify_composition(atol=1e-5), f"S_{n}: composition verification failed"

    def test_s5_composition(self):
        """S_5 composition — separate test because it's slower (14400 pairs)."""
        g = SnRepresentation(5)
        assert g.verify_composition(atol=1e-4)

    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def test_transposition_involution(self, n):
        """All adjacent transpositions satisfy s^2 = e."""
        g = SnRepresentation(n)
        identity = tuple(range(n))
        for i in range(n - 1):
            # Adjacent transposition swapping i and i+1
            sigma = list(range(n))
            sigma[i], sigma[i + 1] = sigma[i + 1], sigma[i]
            idx = g._elem_to_idx[tuple(sigma)]
            R = g.irrep_matrix(idx)
            assert torch.allclose(R @ R, torch.eye(g.total_dim), atol=1e-6), (
                f"S_{n}: transposition s_{i} is not an involution"
            )

    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def test_orthogonality(self, n):
        """All irrep matrices should be orthogonal."""
        g = SnRepresentation(n)
        for idx in range(min(g.order, 24)):  # spot-check first 24 elements
            R = g.irrep_matrix(idx)
            assert torch.allclose(R @ R.T, torch.eye(g.total_dim), atol=1e-5), (
                f"S_{n}: element {idx} is not orthogonal"
            )

    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def test_sign_representation(self, n):
        """The sign irrep block should equal the sign of the permutation."""
        g = SnRepresentation(n)
        # Sign representation is the last partition (1,1,...,1)
        sign_offset = g.total_dim - 1  # last 1x1 block

        for idx, sigma in enumerate(g._elements):
            R = g.irrep_matrix(idx)
            sign_val = R[sign_offset, sign_offset].item()
            # Count inversions to determine sign
            inversions = sum(1 for i in range(n) for j in range(i + 1, n) if sigma[i] > sigma[j])
            expected = (-1.0) ** inversions
            assert abs(sign_val - expected) < 1e-6, (
                f"S_{n}: element {idx} sign={sign_val}, expected={expected}"
            )

    def test_s6_basic(self):
        """S_6 basic properties — not full composition (too slow)."""
        g = SnRepresentation(6)
        assert g.order == 720
        assert g.total_dim == 76
        # Check identity
        identity = tuple(range(6))
        idx = g._elem_to_idx[identity]
        assert torch.allclose(g.irrep_matrix(idx), torch.eye(76), atol=1e-6)
