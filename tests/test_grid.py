"""Tests for gaussian_plume.grid grid helper functions."""

from __future__ import annotations

import numpy as np
import pytest

from gaussian_plume.grid import (
    bin_centres,
    bin_edges,
    cartesian_to_cylindrical,
    cylindrical_grid_coordinates,
    cylindrical_grid_shape,
    cylindrical_grid_size,
    cylindrical_to_cartesian,
    grid_coordinates,
    grid_edges,
    grid_shape,
    grid_shape_from_centres,
    grid_size,
)


class TestBinCentres:
    def test_uniform_edges(self):
        edges = [0.0, 1.0, 2.0, 3.0]
        np.testing.assert_allclose(bin_centres(edges), np.array([0.5, 1.5, 2.5]))

    def test_nonuniform_edges(self):
        edges = [0.0, 1.0, 3.0, 6.0]
        np.testing.assert_allclose(bin_centres(edges), np.array([0.5, 2.0, 4.5]))

    def test_single_edge_returns_empty(self):
        result = bin_centres([1.0])
        assert isinstance(result, np.ndarray)
        assert result.size == 0


class TestBinEdges:
    def test_uniform_centres(self):
        centres = [0.5, 1.5, 2.5]
        np.testing.assert_allclose(bin_edges(centres), np.array([0.0, 1.0, 2.0, 3.0]))

    def test_nonuniform_centres(self):
        centres = [1.0, 2.0, 4.0]
        np.testing.assert_allclose(bin_edges(centres), np.array([0.5, 1.5, 3.0, 5.0]))

    def test_too_few_centres_raises(self):
        with pytest.raises(ValueError, match="At least two centres"):
            bin_edges([1.0])


class TestGridCoordinates:
    def test_2d_coordinates(self):
        x, y, z = grid_coordinates([0.0, 1.0, 2.0], [10.0, 12.0, 16.0])
        np.testing.assert_allclose(x, np.array([0.5, 1.5]))
        np.testing.assert_allclose(y, np.array([11.0, 14.0]))
        assert z is None

    def test_3d_coordinates(self):
        x, y, z = grid_coordinates([0.0, 2.0], [0.0, 4.0], [0.0, 1.0, 3.0])
        np.testing.assert_allclose(x, np.array([1.0]))
        np.testing.assert_allclose(y, np.array([2.0]))
        assert z is not None
        np.testing.assert_allclose(z, np.array([0.5, 2.0]))


class TestGridEdges:
    def test_2d_edges(self):
        x_edges, y_edges, z_edges = grid_edges([0.5, 1.5], [11.0, 14.0])
        np.testing.assert_allclose(x_edges, np.array([0.0, 1.0, 2.0]))
        np.testing.assert_allclose(y_edges, np.array([9.5, 12.5, 15.5]))
        assert z_edges is None

    def test_3d_edges(self):
        x_edges, y_edges, z_edges = grid_edges([1.0, 3.0], [2.0, 5.0], [0.5, 2.0])
        np.testing.assert_allclose(x_edges, np.array([0.0, 2.0, 4.0]))
        np.testing.assert_allclose(y_edges, np.array([0.5, 3.5, 6.5]))
        assert z_edges is not None
        np.testing.assert_allclose(z_edges, np.array([-0.25, 1.25, 2.75]))


class TestGridShape:
    def test_shape_from_edges_2d(self):
        assert grid_shape([0.0, 1.0, 2.0], [0.0, 5.0, 10.0, 15.0]) == (2, 3, None)

    def test_shape_from_edges_3d(self):
        assert grid_shape([0.0, 1.0], [0.0, 2.0], [0.0, 0.5, 1.0]) == (1, 1, 2)

    def test_shape_from_centres_2d(self):
        assert grid_shape_from_centres([0.5, 1.5], [1.0, 2.0, 3.0]) == (2, 3, None)

    def test_shape_from_centres_3d(self):
        assert grid_shape_from_centres([0.5, 1.5], [1.0, 3.0], [0.25, 0.75]) == (2, 2, 2)

    def test_shape_from_centres_requires_two_or_more_per_axis(self):
        with pytest.raises(ValueError, match="At least two centres"):
            grid_shape_from_centres([1.0], [2.0], [0.25, 0.75])


class TestGridSize:
    def test_size_2d(self):
        assert grid_size([0.0, 1.0, 2.0], [0.0, 2.0, 4.0]) == 4

    def test_size_3d(self):
        assert grid_size([0.0, 1.0, 2.0], [0.0, 2.0], [0.0, 1.0, 2.0, 3.0]) == 6


class TestCylindricalGridCoordinates:
    def test_cylindrical_coordinates(self):
        r, phi, z = cylindrical_grid_coordinates(
            [0.0, 1.0, 3.0],
            [0.0, np.pi / 2.0, np.pi],
            [-1.0, 1.0, 5.0],
        )
        np.testing.assert_allclose(r, np.array([0.5, 2.0]))
        np.testing.assert_allclose(phi, np.array([np.pi / 4.0, 3.0 * np.pi / 4.0]))
        np.testing.assert_allclose(z, np.array([0.0, 3.0]))


class TestCylindricalGridShapeAndSize:
    def test_shape(self):
        shape = cylindrical_grid_shape(
            [0.0, 1.0, 2.0, 3.0],
            [0.0, np.pi, 2.0 * np.pi],
            [0.0, 1.0, 2.0],
        )
        assert shape == (3, 2, 2)

    def test_size(self):
        size = cylindrical_grid_size(
            [0.0, 1.0, 2.0, 3.0],
            [0.0, np.pi, 2.0 * np.pi],
            [0.0, 1.0, 2.0],
        )
        assert size == 12


class TestCoordinateTransforms:
    def test_cylindrical_to_cartesian_known_points(self):
        r = np.array([1.0, 2.0, 3.0])
        phi = np.array([0.0, np.pi / 2.0, np.pi])
        z = np.array([4.0, 5.0, 6.0])

        x, y, z_out = cylindrical_to_cartesian(r, phi, z)

        np.testing.assert_allclose(x, np.array([1.0, 0.0, -3.0]), atol=1e-12)
        np.testing.assert_allclose(y, np.array([0.0, 2.0, 0.0]), atol=1e-12)
        np.testing.assert_allclose(z_out, z)

    def test_cartesian_to_cylindrical_known_points(self):
        x = np.array([1.0, 0.0, -1.0, 0.0])
        y = np.array([0.0, 1.0, 0.0, -1.0])
        z = np.array([10.0, 11.0, 12.0, 13.0])

        r, phi, z_out = cartesian_to_cylindrical(x, y, z)

        np.testing.assert_allclose(r, np.array([1.0, 1.0, 1.0, 1.0]))
        np.testing.assert_allclose(
            phi,
            np.array([0.0, np.pi / 2.0, np.pi, -np.pi / 2.0]),
            atol=1e-12,
        )
        np.testing.assert_allclose(z_out, z)

    def test_roundtrip_cylindrical_cartesian(self):
        r_in = np.array([0.25, 1.0, 5.0])
        phi_in = np.array([-2.2, 0.5, 2.8])
        z_in = np.array([-3.0, 0.0, 7.5])

        x, y, z_mid = cylindrical_to_cartesian(r_in, phi_in, z_in)
        r_out, phi_out, z_out = cartesian_to_cylindrical(x, y, z_mid)

        np.testing.assert_allclose(r_out, r_in, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(z_out, z_in, rtol=1e-12, atol=1e-12)
        # Compare angle modulo 2*pi to account for branch-cut representation.
        delta = np.arctan2(np.sin(phi_out - phi_in), np.cos(phi_out - phi_in))
        np.testing.assert_allclose(delta, np.zeros_like(delta), atol=1e-12)
