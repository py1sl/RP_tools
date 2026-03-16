"""Nuclide data class and JSON loader for RP_tools.

This module is part of the ``utilities`` package, which provides shared
data-handling classes and common functions used across all RP_tools tool
packages (Gaussian plume model, skin dose, ingestion dose, etc.).

This module provides:
- :class:`Nuclide` – an immutable data class representing a single nuclide.
- :func:`load_nuclides` – reads ``data/nuclides.json`` and returns a dictionary
  mapping nuclide names to :class:`Nuclide` instances.

Typical usage::

    from utilities.nuclide import load_nuclides

    nuclides = load_nuclides()
    co60 = nuclides["Co60"]
    print(co60.half_life_seconds)   # 166348000.0
    print(co60.gamma_lines)         # [{'energy_MeV': 1.1732, ...}, ...]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Default path to the bundled nuclides data file.
_DEFAULT_DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "nuclides.json"


class Nuclide:
    """Represents a single nuclide with its nuclear properties.

    Attributes:
        name: Short identifier, e.g. ``"Co60"``.
        long_name: Human-readable name, e.g. ``"Cobalt-60"``.
        symbol: Element symbol, e.g. ``"Co"``.
        A: Mass number.
        Z: Atomic number.
        stable: ``True`` if the nuclide is stable.
        half_life_seconds: Half-life in seconds (``None`` for stable nuclides).
        half_life_years: Half-life in years for convenience
            (``None`` for stable nuclides).
        decay_modes: List of decay-mode dicts. Each dict contains at minimum
            ``"mode"`` (str) and ``"branching_fraction"`` (float). Empty list
            for stable nuclides.
        gamma_lines: List of gamma/photon emission line dicts with
            ``"energy_MeV"`` and ``"intensity_percent"`` keys. Empty list for
            stable nuclides.
        x_ray_lines: List of characteristic X-ray line dicts. Empty list when
            not applicable.
        beta_lines: List of beta endpoint dicts with ``"endpoint_energy_MeV"``
            and ``"intensity_percent"`` keys. Empty list when not applicable.
    """

    def __init__(self, data: dict[str, Any]) -> None:
        """Initialise a :class:`Nuclide` from a dictionary of properties.

        Args:
            data: Dictionary as parsed from a single entry in ``nuclides.json``.

        Raises:
            KeyError: If a required field is missing from *data*.
            ValueError: If *data* contains invalid values (e.g. negative A or Z).
        """
        self.name: str = data["name"]
        self.long_name: str = data["long_name"]
        self.symbol: str = data["symbol"]

        self.A: int = int(data["A"])
        self.Z: int = int(data["Z"])
        if self.A <= 0:
            raise ValueError(f"Mass number A must be positive, got {self.A}")
        if self.Z < 0:
            raise ValueError(f"Atomic number Z must be non-negative, got {self.Z}")

        self.stable: bool = bool(data["stable"])

        if not self.stable:
            self.half_life_seconds: float | None = float(data["half_life_seconds"])
            self.half_life_years: float | None = float(data["half_life_years"])
            if self.half_life_seconds <= 0:
                raise ValueError(
                    f"half_life_seconds must be positive for unstable nuclide "
                    f"'{self.name}', got {self.half_life_seconds}"
                )
        else:
            self.half_life_seconds = None
            self.half_life_years = None

        self.decay_modes: list[dict[str, Any]] = list(data.get("decay_modes", []))
        self.gamma_lines: list[dict[str, Any]] = list(data.get("gamma_lines", []))
        self.x_ray_lines: list[dict[str, Any]] = list(data.get("x_ray_lines", []))
        self.beta_lines: list[dict[str, Any]] = list(data.get("beta_lines", []))

    # ------------------------------------------------------------------
    # Derived / convenience properties
    # ------------------------------------------------------------------

    @property
    def N(self) -> int:
        """Neutron number (A − Z)."""
        return self.A - self.Z

    def __repr__(self) -> str:
        stability = "stable" if self.stable else f"T½={self.half_life_years:.4g} y"
        return f"Nuclide({self.name!r}, Z={self.Z}, A={self.A}, {stability})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Nuclide):
            return NotImplemented
        return self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)


def load_nuclides(
    data_file: str | Path | None = None,
) -> dict[str, Nuclide]:
    """Load nuclides from a JSON file and return a dictionary of :class:`Nuclide` objects.

    Args:
        data_file: Path to a ``nuclides.json``-formatted file. When ``None``
            (default), the bundled ``data/nuclides.json`` file is used.

    Returns:
        A dictionary mapping each nuclide name (e.g. ``"Co60"``) to the
        corresponding :class:`Nuclide` instance.

    Raises:
        FileNotFoundError: If *data_file* does not exist.
        ValueError: If the JSON structure is invalid or a nuclide entry
            contains bad values.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    path = Path(data_file) if data_file is not None else _DEFAULT_DATA_FILE

    if not path.exists():
        raise FileNotFoundError(f"Nuclide data file not found: {path}")

    with path.open(encoding="utf-8") as fh:
        raw = json.load(fh)

    if "nuclides" not in raw:
        raise ValueError(
            f"Expected a top-level 'nuclides' key in {path}; got keys: "
            + ", ".join(raw.keys())
        )

    result: dict[str, Nuclide] = {}
    for name, entry in raw["nuclides"].items():
        result[name] = Nuclide(entry)

    return result
