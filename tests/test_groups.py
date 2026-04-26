"""Tests for group representations."""

import torch
import pytest

from src.groups import Z2Representation, Z3Representation, S2Representation, S3Representation


class TestZ2:
    def test_order(self):
        g = Z2Representation()
        assert g.order == 2

    def test_total_dim(self):
        g = Z2Representation()
        assert g.total_dim == 2

    def test_identity(self):
        g = Z2Representation()
        assert torch.allclose(g.irrep_matrix(0), torch.eye(2))

    def test_involution(self):
        g = Z2Representation()
        R = g.irrep_matrix(1)
        assert torch.allclose(R @ R, torch.eye(2), atol=1e-6)

    def test_composition(self):
        g = Z2Representation()
        assert g.verify_composition()


class TestZ3:
    def test_order(self):
        g = Z3Representation()
        assert g.order == 3

    def test_total_dim(self):
        g = Z3Representation()
        assert g.total_dim == 3  # 1 + 2

    def test_identity(self):
        g = Z3Representation()
        assert torch.allclose(g.irrep_matrix(0), torch.eye(3))

    def test_r_cubed_is_identity(self):
        g = Z3Representation()
        R = g.irrep_matrix(1)
        assert torch.allclose(R @ R @ R, torch.eye(3), atol=1e-6)

    def test_r2_is_r_squared(self):
        g = Z3Representation()
        R = g.irrep_matrix(1)
        R2 = g.irrep_matrix(2)
        assert torch.allclose(R @ R, R2, atol=1e-6)

    def test_composition(self):
        g = Z3Representation()
        assert g.verify_composition()


class TestS2:
    def test_order(self):
        g = S2Representation()
        assert g.order == 2

    def test_composition(self):
        g = S2Representation()
        assert g.verify_composition()


class TestS3:
    def test_order(self):
        g = S3Representation()
        assert g.order == 6

    def test_total_dim(self):
        g = S3Representation()
        assert g.total_dim == 4  # 1 + 1 + 2

    def test_identity(self):
        g = S3Representation()
        assert torch.allclose(g.irrep_matrix(0), torch.eye(4))

    def test_composition(self):
        """R(g1) @ R(g2) == R(g1*g2) for all pairs."""
        g = S3Representation()
        assert g.verify_composition()

    def test_transposition_involution(self):
        """All transpositions satisfy s^2 = e."""
        g = S3Representation()
        for idx in [1, 2, 3]:  # three transpositions
            R = g.irrep_matrix(idx)
            assert torch.allclose(R @ R, torch.eye(4), atol=1e-6), \
                f"Element {idx} ({g.element_names()[idx]}) is not an involution"

    def test_three_cycle_cubes_to_identity(self):
        """r^3 = e for both 3-cycles."""
        g = S3Representation()
        for idx in [4, 5]:  # two 3-cycles
            R = g.irrep_matrix(idx)
            assert torch.allclose(R @ R @ R, torch.eye(4), atol=1e-6), \
                f"Element {idx} ({g.element_names()[idx]})^3 != e"

    def test_sign_irrep(self):
        """Sign irrep: +1 for even, -1 for odd."""
        g = S3Representation()
        parity = [0, 1, 1, 1, 0, 0]
        for idx in range(6):
            R = g.irrep_matrix(idx)
            sign_val = R[1, 1].item()
            expected = 1.0 if parity[idx] == 0 else -1.0
            assert abs(sign_val - expected) < 1e-6, \
                f"Element {idx}: sign={sign_val}, expected={expected}"

    def test_determinants(self):
        """det = +1 for even, -1 for odd (in the standard 2D irrep)."""
        g = S3Representation()
        parity = [0, 1, 1, 1, 0, 0]
        for idx in range(6):
            R = g.irrep_matrix(idx)
            std_block = R[2:4, 2:4]
            det = torch.det(std_block).item()
            expected = 1.0 if parity[idx] == 0 else -1.0
            assert abs(det - expected) < 1e-5, \
                f"Element {idx}: det={det}, expected={expected}"

    def test_multiplication_table_closure(self):
        """Every product maps to a valid element."""
        g = S3Representation()
        table = g.multiplication_table()
        assert table.min() >= 0
        assert table.max() < g.order
