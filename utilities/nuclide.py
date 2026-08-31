"""Nuclide data class and JSON loader for RP_tools.

This module is part of the ``utilities`` package, which provides shared
data-handling classes and common functions used across all RP_tools tool
packages (Gaussian plume model, skin dose, ingestion dose, etc.).

This module provides:
- :class:`Nuclide` – an immutable data class representing a single nuclide.
- :func:`load_nuclides` – reads ``data/nuclides.json`` and returns a dictionary
  mapping nuclide names to :class:`Nuclide` instances.
- :func:`normalize_nuclide_name` – converts common nuclide string formats to
  canonical ``SymbolA`` form.
- :func:`nuclides_of_element` – returns all loaded nuclides for a given element
  symbol.
- :func:`stable_daughters` – walks all decay branches and returns the set of
  stable end-product nuclides.
- :func:`is_in_chain` – checks whether a nuclide appears anywhere in a parent's
  decay chain.

Typical usage::

    from utilities.nuclide import load_nuclides

    nuclides = load_nuclides()
    co60 = nuclides["Co60"]
    print(co60.half_life_seconds)   # 166348000.0
    print(co60.gamma_lines)         # [{'energy_MeV': 1.1732, ...}, ...]
"""

from __future__ import annotations

import json
import re
from collections import deque
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

    @property
    def element_name(self) -> str:
        """Full element name, e.g. ``"Cobalt"`` for Co-60.

        Derived from :attr:`long_name` by stripping the trailing ``"-A"`` mass
        suffix.  For nuclides whose :attr:`long_name` contains no hyphen (e.g.
        ``"Tritium"``), the full :attr:`long_name` is returned unchanged.
        """
        return self.long_name.rsplit("-", 1)[0]

    @property
    def zaid(self) -> int:
        """ZAID identifier (Z * 1000 + A), e.g. ``27060`` for Co-60."""
        return self.Z * 1000 + self.A

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


def normalize_nuclide_name(name: str) -> str:
    """Normalize common nuclide string formats to canonical ``SymbolA`` form.

    Examples:
        ``"U-235"``, ``"U235"``, and ``"235U"`` all normalize to ``"U235"``.
    """
    raw = name.strip()
    if not raw:
        raise ValueError("Nuclide name cannot be empty.")

    patterns = (
        r"^([A-Za-z]{1,2})(?:[\s\-_])?([0-9]{1,3})$",
        r"^([0-9]{1,3})(?:[\s\-_])?([A-Za-z]{1,2})$",
    )

    symbol: str | None = None
    mass_str: str | None = None

    for pattern in patterns:
        match = re.match(pattern, raw)
        if match is None:
            continue
        left, right = match.groups()
        if left[0].isdigit():
            mass_str, symbol = left, right
        else:
            symbol, mass_str = left, right
        break

    if symbol is None or mass_str is None:
        raise ValueError(
            f"Invalid nuclide name format {name!r}. "
            "Expected forms like 'U-235', 'U235', or '235U'."
        )

    mass_number = int(mass_str)
    if mass_number <= 0:
        raise ValueError(f"Mass number must be positive, got {mass_number}.")

    normalized_symbol = symbol[0].upper() + symbol[1:].lower()
    return f"{normalized_symbol}{mass_number}"


# ---------------------------------------------------------------------------
# Nuclide utility functions
# ---------------------------------------------------------------------------


def _build_za_index(nuclide_db: dict[str, "Nuclide"]) -> dict[tuple[int, int], "Nuclide"]:
    """Return a ``(Z, A) → Nuclide`` mapping for fast lookup by nuclear numbers."""
    return {(n.Z, n.A): n for n in nuclide_db.values()}


def nuclides_of_element(
    symbol: str,
    nuclide_db: dict[str, "Nuclide"],
) -> list["Nuclide"]:
    """Return all nuclides in *nuclide_db* whose element symbol matches *symbol*.

    The comparison is case-insensitive; the symbol is normalised to title-case
    (first letter upper, remainder lower) before matching.

    Args:
        symbol: Element symbol to search for, e.g. ``"Co"`` or ``"co"``.
        nuclide_db: Dictionary mapping nuclide names to
            :class:`Nuclide` instances, as returned by :func:`load_nuclides`.

    Returns:
        List of :class:`Nuclide` instances whose :attr:`~Nuclide.symbol`
        matches *symbol*, sorted by ascending mass number *A*.

    Raises:
        ValueError: If *symbol* is empty.

    Examples::

        nuclides = load_nuclides()
        cobalt_isotopes = nuclides_of_element("Co", nuclides)
        # [Nuclide('Co59', ...), Nuclide('Co60', ...), ...]
    """
    if not symbol.strip():
        raise ValueError("Element symbol cannot be empty.")

    normalised = symbol.strip()[0].upper() + symbol.strip()[1:].lower()
    return sorted(
        (n for n in nuclide_db.values() if n.symbol == normalised),
        key=lambda n: n.A,
    )


def stable_daughters(
    nuclide: "Nuclide",
    nuclide_db: dict[str, "Nuclide"],
    max_steps: int = 200,
) -> set["Nuclide"]:
    """Return the set of stable nuclides that terminate *nuclide*'s decay chain.

    All decay branches recorded in :attr:`~Nuclide.decay_modes` are followed
    simultaneously so that every possible stable end-product is captured, even
    for nuclides with significant branching (e.g. ``beta+`` / ``electron_capture``
    competition).

    If a daughter referenced in the data is absent from *nuclide_db*, the
    traversal silently stops at that point (the missing nuclide is assumed to
    be outside the scope of the loaded database).  Unknown decay-mode strings
    that have no associated nuclear-change rule are skipped similarly.

    Args:
        nuclide: Starting :class:`Nuclide`.
        nuclide_db: Dictionary mapping nuclide names to
            :class:`Nuclide` instances, as returned by :func:`load_nuclides`.
        max_steps: Maximum BFS/DFS depth before giving up (guards against
            malformed data).  Defaults to 200.

    Returns:
        Set of stable :class:`Nuclide` objects reachable from *nuclide*.
        Returns a set containing *nuclide* itself when it is already stable.

    Raises:
        ValueError: If *max_steps* is exceeded.

    Examples::

        nuclides = load_nuclides()
        ends = stable_daughters(nuclides["U238"], nuclides)
        # {Nuclide('Pb206', ...)}
    """
    # Lazy import to avoid a circular dependency between nuclide ↔ radioactive_decay.
    from utilities.radioactive_decay import _DECAY_MODE_DELTAS  # noqa: PLC0415

    if nuclide.stable:
        return {nuclide}

    # Pre-build (Z, A) → Nuclide index.
    za_index = _build_za_index(nuclide_db)

    stable: set[Nuclide] = set()
    # BFS queue of (nuclide, depth).
    queue: deque[tuple[Nuclide, int]] = deque([(nuclide, 0)])
    visited: set[str] = {nuclide.name}

    while queue:
        current, depth = queue.popleft()

        if current.stable:
            stable.add(current)
            continue

        if depth >= max_steps:
            raise ValueError(
                f"stable_daughters: exceeded {max_steps} steps starting from "
                f"{nuclide.name!r}."
            )

        for mode_info in current.decay_modes:
            mode = mode_info["mode"].lower()
            daughter_name: str | None = mode_info.get("daughter")

            if daughter_name is not None:
                if daughter_name not in nuclide_db:
                    continue
                daughter = nuclide_db[daughter_name]
            else:
                if mode not in _DECAY_MODE_DELTAS:
                    continue
                dZ, dA = _DECAY_MODE_DELTAS[mode]
                key = (current.Z + dZ, current.A + dA)
                if key not in za_index:
                    continue
                daughter = za_index[key]

            if daughter.name not in visited:
                visited.add(daughter.name)
                queue.append((daughter, depth + 1))

    return stable


def is_in_chain(
    parent: "Nuclide",
    candidate: "Nuclide",
    nuclide_db: dict[str, "Nuclide"],
    max_steps: int = 200,
) -> bool:
    """Return ``True`` if *candidate* appears anywhere in *parent*'s decay chain.

    All decay branches are explored (same BFS strategy as
    :func:`stable_daughters`), so branching chains are handled correctly.
    *parent* itself is **not** considered to be in its own chain (i.e. the
    function returns ``False`` when ``parent == candidate``).

    Args:
        parent: The nuclide whose decay chain is searched.
        candidate: The nuclide to search for.
        nuclide_db: Dictionary mapping nuclide names to
            :class:`Nuclide` instances, as returned by :func:`load_nuclides`.
        max_steps: Maximum BFS depth (guards against malformed data).
            Defaults to 200.

    Returns:
        ``True`` if *candidate* is a daughter (direct or indirect) of *parent*;
        ``False`` otherwise.

    Raises:
        ValueError: If *max_steps* is exceeded.

    Examples::

        nuclides = load_nuclides()
        print(is_in_chain(nuclides["U238"], nuclides["Pb206"], nuclides))  # True
        print(is_in_chain(nuclides["Co60"], nuclides["Cs137"], nuclides))  # False
    """
    # Lazy import to avoid a circular dependency.
    from utilities.radioactive_decay import _DECAY_MODE_DELTAS  # noqa: PLC0415

    if parent.stable:
        return False

    za_index = _build_za_index(nuclide_db)

    queue: deque[tuple[Nuclide, int]] = deque([(parent, 0)])
    visited: set[str] = {parent.name}

    while queue:
        current, depth = queue.popleft()

        if current.stable:
            continue

        if depth >= max_steps:
            raise ValueError(
                f"is_in_chain: exceeded {max_steps} steps starting from "
                f"{parent.name!r}."
            )

        for mode_info in current.decay_modes:
            mode = mode_info["mode"].lower()
            daughter_name: str | None = mode_info.get("daughter")

            if daughter_name is not None:
                if daughter_name not in nuclide_db:
                    continue
                daughter = nuclide_db[daughter_name]
            else:
                if mode not in _DECAY_MODE_DELTAS:
                    continue
                dZ, dA = _DECAY_MODE_DELTAS[mode]
                key = (current.Z + dZ, current.A + dA)
                if key not in za_index:
                    continue
                daughter = za_index[key]

            if daughter == candidate:
                return True

            if daughter.name not in visited:
                visited.add(daughter.name)
                queue.append((daughter, depth + 1))

    return False
