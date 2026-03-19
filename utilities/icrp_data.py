"""ICRP dose-coefficient table loader utilities.

This module provides lightweight data classes for loading ICRP external dose
coefficient tables distributed in ``data/icrp74`` and ``data/icrp116``.

The source text files are simple whitespace-delimited tables with:
- a one-line description,
- a header row (energy and irradiation geometry columns), and
- numeric data rows.

Typical usage::

    from utilities.icrp_data import ICRPDataLibrary

    lib = ICRPDataLibrary()
    photons = lib.get_table("116", "photons")
    energies = photons.energies_MeV
    ap = photons.column("AP")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import numpy as np

_DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_NUMERIC_START = re.compile(r"^[+-]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[Ee][+-]?\d+)?$")


@dataclass(frozen=True)
class ICRPTable:
    """Single ICRP dose-coefficient table.

    Attributes:
        publication: ICRP publication identifier (e.g. ``"74"`` or ``"116"``).
        particle: Particle key inferred from filename stem (e.g. ``"photons"``).
        description: Free-text description from the first line of the file.
        columns: Column names in table order.
        values: Numeric array of shape ``(n_rows, n_columns)``.
    """

    publication: str
    particle: str
    description: str
    columns: tuple[str, ...]
    values: np.ndarray

    @property
    def energies_MeV(self) -> np.ndarray:
        """Return the energy column (MeV)."""
        return self.values[:, 0]

    def column(self, name: str) -> np.ndarray:
        """Return a named column as a 1D array.

        Args:
            name: Column name from :attr:`columns`.

        Raises:
            KeyError: If the column does not exist.
        """
        try:
            idx = self.columns.index(name)
        except ValueError as exc:
            raise KeyError(
                f"Column {name!r} not found in table {self.publication}/{self.particle}. "
                f"Available: {', '.join(self.columns)}"
            ) from exc
        return self.values[:, idx]


class ICRPDataLibrary:
    """In-memory class-based store of ICRP external dose tables."""

    def __init__(self, data_dir: str | Path | None = None) -> None:
        self.data_dir = Path(data_dir) if data_dir is not None else _DEFAULT_DATA_DIR
        self._tables: dict[tuple[str, str], ICRPTable] = {}
        self._load_all()

    def publications(self) -> tuple[str, ...]:
        """Return loaded publication IDs."""
        return tuple(sorted({pub for pub, _ in self._tables}))

    def particles(self, publication: str | None = None) -> tuple[str, ...]:
        """Return available particle names.

        Args:
            publication: Optional publication filter (e.g. ``"74"`` or ``"116"``).
        """
        if publication is None:
            items = (particle for _, particle in self._tables)
        else:
            pub = _normalize_publication(publication)
            items = (particle for p, particle in self._tables if p == pub)
        return tuple(sorted(set(items)))

    def get_table(self, publication: str, particle: str) -> ICRPTable:
        """Return one table by publication and particle.

        Raises:
            KeyError: If the requested table is not available.
        """
        key = (_normalize_publication(publication), particle)
        if key not in self._tables:
            pub, part = key
            raise KeyError(
                f"ICRP table not found for publication {pub!r}, particle {part!r}. "
                f"Available particles for {pub}: {', '.join(self.particles(pub))}"
            )
        return self._tables[key]

    def tables_for_publication(self, publication: str) -> dict[str, ICRPTable]:
        """Return all tables for one publication keyed by particle name."""
        pub = _normalize_publication(publication)
        return {
            particle: table
            for (p, particle), table in self._tables.items()
            if p == pub
        }

    def _load_all(self) -> None:
        for publication in ("74", "116"):
            folder = self.data_dir / f"icrp{publication}"
            if not folder.exists():
                raise FileNotFoundError(f"ICRP data folder not found: {folder}")
            for file_path in sorted(folder.glob("*.txt")):
                table = _parse_icrp_text_table(file_path, publication)
                self._tables[(publication, table.particle)] = table


def load_icrp_data(data_dir: str | Path | None = None) -> ICRPDataLibrary:
    """Load ICRP-74 and ICRP-116 external dose coefficient tables."""
    return ICRPDataLibrary(data_dir=data_dir)


def _normalize_publication(publication: str) -> str:
    pub = str(publication).strip()
    if pub.lower().startswith("icrp"):
        pub = pub[4:]
    return pub


def _parse_header_tokens(line: str) -> tuple[str, ...]:
    tokens = line.split()
    if len(tokens) < 2:
        raise ValueError(f"Header row is malformed: {line!r}")

    # Most files use "Energy (MeV)" as the first two tokens.
    if len(tokens) >= 2 and tokens[0].lower() == "energy" and tokens[1].startswith("("):
        return ("Energy (MeV)", *tokens[2:])

    return tuple(tokens)


def _iter_numeric_rows(lines: Iterable[str]) -> Iterable[tuple[int, str]]:
    for line_no, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line:
            continue
        first = line.split()[0]
        if _NUMERIC_START.match(first):
            yield line_no, line


def _parse_icrp_text_table(file_path: Path, publication: str) -> ICRPTable:
    with file_path.open(encoding="utf-8") as fh:
        lines = fh.readlines()

    if not lines:
        raise ValueError(f"ICRP table file is empty: {file_path}")

    description = lines[0].strip()
    header_line = None
    for raw in lines[1:]:
        stripped = raw.strip()
        if stripped.lower().startswith("energy"):
            header_line = stripped
            break
    if header_line is None:
        raise ValueError(f"Could not find header row in ICRP table: {file_path}")

    columns = _parse_header_tokens(header_line)
    rows: list[list[float]] = []

    for line_no, line in _iter_numeric_rows(lines):
        parts = line.split()
        if len(parts) != len(columns):
            raise ValueError(
                f"Unexpected column count in {file_path} at line {line_no}: "
                f"expected {len(columns)}, got {len(parts)}"
            )
        rows.append([float(v) for v in parts])

    if not rows:
        raise ValueError(f"No numeric data rows found in ICRP table: {file_path}")

    return ICRPTable(
        publication=publication,
        particle=file_path.stem,
        description=description,
        columns=columns,
        values=np.asarray(rows, dtype=float),
    )
