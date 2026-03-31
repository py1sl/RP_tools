"""Atmospheric dispersion coefficients for the NRPB-R91 Gaussian plume model.

Implements the Clarke (1979) σy and σz parametrisation used in NRPB-R91
(Simmonds et al., 1993) for six Pasquill-Gifford atmospheric stability
categories.  All distances are in metres.

The dispersion coefficient formula is::

    σ(x) = a · x · (1 + b · x)^c

where x is the downwind distance in metres and σ is in metres.

Reference:
    Simmonds, J. R. et al. (1993). *The Methodology for Assessing the
    Radiological Consequences of Routine Releases of Radionuclides to the
    Atmosphere*.  NRPB-R91, National Radiological Protection Board, Chilton, UK.
"""

from __future__ import annotations

from typing import NamedTuple, Union

import numpy as np

ArrayOrFloat = Union[float, np.ndarray]

# ---------------------------------------------------------------------------
# Stability category definitions
# ---------------------------------------------------------------------------

# Pasquill-Gifford stability categories used in NRPB-R91, ordered from
# most unstable (A) to most stable (F).
STABILITY_CATEGORIES: tuple[str, ...] = ("A", "B", "C", "D", "E", "F")

# Human-readable descriptions of each stability category.
CATEGORY_DESCRIPTIONS: dict[str, str] = {
    "A": "Very unstable",
    "B": "Unstable",
    "C": "Slightly unstable",
    "D": "Neutral",
    "E": "Slightly stable",
    "F": "Moderately stable",
}

# ---------------------------------------------------------------------------
# Clarke (1979) coefficient table
# ---------------------------------------------------------------------------


class _DispersionCoeffs(NamedTuple):
    """Clarke (1979) power-law parameters for one stability category.

    Both σy and σz are evaluated with the same formula::

        σ(x) = a · x · (1 + b · x)^c

    where x is in metres and σ is in metres.
    """

    a_y: float  # σy multiplicative constant
    b_y: float  # σy denominator coefficient
    c_y: float  # σy exponent
    a_z: float  # σz multiplicative constant
    b_z: float  # σz denominator coefficient
    c_z: float  # σz exponent


# Clarke (1979) parameters for open-country (rural) conditions as reproduced
# in NRPB-R91.  Categories A–F correspond to Pasquill-Gifford stability
# classes from very unstable (A) to moderately stable (F).
_COEFFS: dict[str, _DispersionCoeffs] = {
    "A": _DispersionCoeffs(a_y=0.22,  b_y=0.0001, c_y=-0.5, a_z=0.20,  b_z=0.0,    c_z=0.0),
    "B": _DispersionCoeffs(a_y=0.16,  b_y=0.0001, c_y=-0.5, a_z=0.12,  b_z=0.0,    c_z=0.0),
    "C": _DispersionCoeffs(a_y=0.11,  b_y=0.0001, c_y=-0.5, a_z=0.08,  b_z=0.0002, c_z=-0.5),
    "D": _DispersionCoeffs(a_y=0.08,  b_y=0.0001, c_y=-0.5, a_z=0.06,  b_z=0.0015, c_z=-0.5),
    "E": _DispersionCoeffs(a_y=0.06,  b_y=0.0001, c_y=-0.5, a_z=0.03,  b_z=0.0003, c_z=-1.0),
    "F": _DispersionCoeffs(a_y=0.04,  b_y=0.0001, c_y=-0.5, a_z=0.016, b_z=0.0003, c_z=-1.0),
}

# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def sigma_y(x: ArrayOrFloat, category: str) -> ArrayOrFloat:
    """Return the crosswind (horizontal) dispersion coefficient σy in metres.

    Uses the Clarke (1979) parametrisation: ``σy = a·x·(1 + b·x)^c``.

    Args:
        x: Downwind distance from the source (m).  Must be positive.  May be a
            scalar or a NumPy array; the returned value has the same type/shape.
        category: Pasquill-Gifford stability category, one of ``'A'``–``'F'``.

    Returns:
        σy in metres (scalar or array matching *x*).

    Raises:
        ValueError: If any value of *x* ≤ 0 or *category* is not recognised.
    """
    _validate(x, category)
    p = _COEFFS[category]
    return p.a_y * x * (1.0 + p.b_y * x) ** p.c_y


def sigma_z(x: ArrayOrFloat, category: str) -> ArrayOrFloat:
    """Return the vertical dispersion coefficient σz in metres.

    Uses the Clarke (1979) parametrisation: ``σz = a·x·(1 + b·x)^c``.

    Args:
        x: Downwind distance from the source (m).  Must be positive.  May be a
            scalar or a NumPy array; the returned value has the same type/shape.
        category: Pasquill-Gifford stability category, one of ``'A'``–``'F'``.

    Returns:
        σz in metres (scalar or array matching *x*).

    Raises:
        ValueError: If any value of *x* ≤ 0 or *category* is not recognised.
    """
    _validate(x, category)
    p = _COEFFS[category]
    return p.a_z * x * (1.0 + p.b_z * x) ** p.c_z


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate(x: ArrayOrFloat, category: str) -> None:
    """Raise ValueError if *x* or *category* are invalid.

    Accepts both scalars and NumPy arrays for *x*.
    """
    x_arr = np.asarray(x)
    if x_arr.ndim == 0:
        if float(x_arr) <= 0:
            raise ValueError(f"Downwind distance x must be positive, got {float(x_arr)}")
    elif np.any(x_arr <= 0):
        count = int(np.sum(x_arr <= 0))
        raise ValueError(
            f"All downwind distances x must be positive; "
            f"found {count} value(s) ≤ 0."
        )
    if category not in _COEFFS:
        raise ValueError(
            f"Unknown stability category {category!r}. "
            f"Valid options are: {', '.join(STABILITY_CATEGORIES)}"
        )
