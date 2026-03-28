"""Tools for creating and working with a grid of points in 2D or 3D space."""

import numpy as np


def bin_centres(edges: list[float]) -> np.ndarray:
    """Return the midpoints between consecutive bin edges."""
    arr = np.asarray(edges, dtype=float)
    return 0.5 * (arr[:-1] + arr[1:])


def bin_edges(centres: list[float]) -> np.ndarray:
    """Return the edges of bins given their centres."""
    arr = np.asarray(centres, dtype=float)
    if len(arr) < 2:
        raise ValueError("At least two centres are required to calculate edges.")
    # Calculate the differences between consecutive centres
    diffs = np.diff(arr)
    # Calculate the edges by subtracting half the difference from the first centre
    # and adding half the difference to the last centre
    edges = np.empty(len(arr) + 1)
    edges[1:-1] = arr[:-1] + diffs / 2
    edges[0] = arr[0] - diffs[0] / 2
    edges[-1] = arr[-1] + diffs[-1] / 2
    return edges


def grid_coordinates(
    x_edges: list[float],
    y_edges: list[float],
    z_edges: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Return the coordinates of the centres of a grid defined by the given edges."""
    x_centres = bin_centres(x_edges)
    y_centres = bin_centres(y_edges)
    z_centres = None
    if z_edges is not None:
        z_centres = bin_centres(z_edges)
    return x_centres, y_centres, z_centres


def grid_edges(
    x_centres: list[float],
    y_centres: list[float],
    z_centres: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Return the edges of a grid defined by the given centres."""
    x_edges = bin_edges(x_centres)
    y_edges = bin_edges(y_centres)
    z_edges = None
    if z_centres is not None:
        z_edges = bin_edges(z_centres)
    return x_edges, y_edges, z_edges


def grid_shape(x_edges: list[float], y_edges: list[float], z_edges: list[float] | None = None) -> tuple[int, int, int | None]:
    """Return the shape of a grid defined by the given edges."""
    x_centres = bin_centres(x_edges)
    y_centres = bin_centres(y_edges)
    z_centres = None
    if z_edges is not None:
        z_centres = bin_centres(z_edges)
    return len(x_centres), len(y_centres), len(z_centres) if z_centres is not None else None


def grid_shape_from_centres(
    x_centres: list[float],
    y_centres: list[float],
    z_centres: list[float] | None = None,
) -> tuple[int, int, int | None]:
    """Return the shape of a grid defined by the given centres."""
    if len(x_centres) < 2:
        raise ValueError("At least two centres are required to calculate edges.")
    if len(y_centres) < 2:
        raise ValueError("At least two centres are required to calculate edges.")
    if z_centres is not None and len(z_centres) < 2:
        raise ValueError("At least two centres are required to calculate edges.")
    return len(x_centres), len(y_centres), len(z_centres) if z_centres is not None else None


def grid_size(x_edges: list[float], y_edges: list[float], z_edges: list[float] | None = None) -> int:
    """Return the total number of points in a grid defined by the given edges."""
    x_centres = bin_centres(x_edges)
    y_centres = bin_centres(y_edges)
    z_centres = None
    if z_edges is not None:
        z_centres = bin_centres(z_edges)
    return len(x_centres) * len(y_centres) * (len(z_centres) if z_centres is not None else 1)


def cylindrical_grid_coordinates(
    r_edges: list[float],
    phi_edges: list[float],
    z_edges: list[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the coordinates of the centres of a cylindrical grid defined by the given edges."""
    r_centres = bin_centres(r_edges)
    phi_centres = bin_centres(phi_edges)
    z_centres = bin_centres(z_edges)
    return r_centres, phi_centres, z_centres


def cylindrical_grid_shape(
    r_edges: list[float],
    phi_edges: list[float],
    z_edges: list[float],
) -> tuple[int, int, int]:
    """Return the shape of a cylindrical grid defined by the given edges."""
    r_centres = bin_centres(r_edges)
    phi_centres = bin_centres(phi_edges)
    z_centres = bin_centres(z_edges)
    return len(r_centres), len(phi_centres), len(z_centres)


def cylindrical_grid_size(
    r_edges: list[float],
    phi_edges: list[float],
    z_edges: list[float],
) -> int:
    """Return the total number of points in a cylindrical grid defined by the given edges."""
    r_centres = bin_centres(r_edges)
    phi_centres = bin_centres(phi_edges)
    z_centres = bin_centres(z_edges)
    return len(r_centres) * len(phi_centres) * len(z_centres)


def cylindrical_to_cartesian(
    r: np.ndarray,
    phi: np.ndarray,
    z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert cylindrical coordinates to Cartesian coordinates."""
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return x, y, z


def cartesian_to_cylindrical(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert Cartesian coordinates to cylindrical coordinates."""
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return r, phi, z
