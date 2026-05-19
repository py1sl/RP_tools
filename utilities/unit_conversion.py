"""Radiation-related unit conversion helpers.

This module provides lightweight conversions for common radiation protection
quantities:

* Activity: becquerel (Bq) and curie (Ci)
* Dose equivalent / effective dose: sievert (Sv) and rem
* Absorbed dose: gray (Gy) and rad

Standard SI prefixes are supported for unit symbols, such as ``mSv``,
``uCi``, ``MBq``, and ``mrad``.
"""

from __future__ import annotations

from dataclasses import dataclass


SI_PREFIX_FACTORS: dict[str, float] = {
    "p": 1e-12,
    "n": 1e-9,
    "u": 1e-6,
    "m": 1e-3,
    "c": 1e-2,
    "d": 1e-1,
    "": 1.0,
    "da": 1e1,
    "h": 1e2,
    "k": 1e3,
    "M": 1e6,
    "G": 1e9,
    "T": 1e12,
}

UNIT_SYMBOLS: dict[str, tuple[str, float]] = {
    "Sv": ("dose_equivalent", 1.0),
    "rem": ("dose_equivalent", 1.0e-2),
    "Gy": ("absorbed_dose", 1.0),
    "rad": ("absorbed_dose", 1.0e-2),
    "Bq": ("activity", 1.0),
    "Ci": ("activity", 3.7e10),
}

UNIT_NAME_ALIASES: dict[str, tuple[str, float]] = {
    "sievert": ("dose_equivalent", 1.0),
    "sieverts": ("dose_equivalent", 1.0),
    "sv": ("dose_equivalent", 1.0),
    "rem": ("dose_equivalent", 1.0e-2),
    "rems": ("dose_equivalent", 1.0e-2),
    "gray": ("absorbed_dose", 1.0),
    "grays": ("absorbed_dose", 1.0),
    "gy": ("absorbed_dose", 1.0),
    "rad": ("absorbed_dose", 1.0e-2),
    "rads": ("absorbed_dose", 1.0e-2),
    "becquerel": ("activity", 1.0),
    "becquerels": ("activity", 1.0),
    "bq": ("activity", 1.0),
    "curie": ("activity", 3.7e10),
    "curies": ("activity", 3.7e10),
    "ci": ("activity", 3.7e10),
}


@dataclass(frozen=True)
class ParsedUnit:
    """Normalized representation of a supported unit."""

    family: str
    factor_to_base: float


def _normalize_symbol(unit: str) -> str:
    return unit.strip().replace("µ", "u").replace("μ", "u")


def _parse_unit(unit: str) -> ParsedUnit:
    normalized = _normalize_symbol(unit)
    if not normalized:
        raise ValueError("unit must be a non-empty string")

    alias_key = normalized.lower()
    if alias_key in UNIT_NAME_ALIASES:
        family, factor_to_base = UNIT_NAME_ALIASES[alias_key]
        return ParsedUnit(family=family, factor_to_base=factor_to_base)

    for base_symbol in sorted(UNIT_SYMBOLS, key=len, reverse=True):
        if normalized == base_symbol:
            family, factor_to_base = UNIT_SYMBOLS[base_symbol]
            return ParsedUnit(family=family, factor_to_base=factor_to_base)

        if normalized.endswith(base_symbol):
            prefix = normalized[: -len(base_symbol)]
            if prefix in SI_PREFIX_FACTORS:
                family, base_factor = UNIT_SYMBOLS[base_symbol]
                return ParsedUnit(
                    family=family,
                    factor_to_base=SI_PREFIX_FACTORS[prefix] * base_factor,
                )

    raise ValueError(f"unsupported unit: {unit}")


def convert_radiation_unit(value: float, from_unit: str, to_unit: str) -> float:
    """Convert between supported radiation-related units.

    Args:
        value: Numeric value to convert.
        from_unit: Source unit symbol or name.
        to_unit: Target unit symbol or name.

    Returns:
        Converted value in ``to_unit``.

    Raises:
        ValueError: If either unit is unsupported or if the units belong to
            different quantity families.
    """
    from_parsed = _parse_unit(from_unit)
    to_parsed = _parse_unit(to_unit)

    if from_parsed.family != to_parsed.family:
        raise ValueError(
            f"cannot convert {from_unit} ({from_parsed.family}) "
            f"to {to_unit} ({to_parsed.family})"
        )

    value_in_base = value * from_parsed.factor_to_base
    return value_in_base / to_parsed.factor_to_base


def convert_activity(value: float, from_unit: str, to_unit: str) -> float:
    """Convert between supported activity units."""
    _validate_family(from_unit, expected_family="activity")
    _validate_family(to_unit, expected_family="activity")
    return convert_radiation_unit(value, from_unit, to_unit)


def convert_dose_equivalent(value: float, from_unit: str, to_unit: str) -> float:
    """Convert between supported dose equivalent / effective dose units."""
    _validate_family(from_unit, expected_family="dose_equivalent")
    _validate_family(to_unit, expected_family="dose_equivalent")
    return convert_radiation_unit(value, from_unit, to_unit)


def convert_absorbed_dose(value: float, from_unit: str, to_unit: str) -> float:
    """Convert between supported absorbed dose units."""
    _validate_family(from_unit, expected_family="absorbed_dose")
    _validate_family(to_unit, expected_family="absorbed_dose")
    return convert_radiation_unit(value, from_unit, to_unit)


def prefix_factor(prefix: str) -> float:
    """Return the scale factor for a supported SI prefix."""
    normalized = _normalize_symbol(prefix)
    if normalized not in SI_PREFIX_FACTORS:
        raise ValueError(f"unsupported SI prefix: {prefix}")
    return SI_PREFIX_FACTORS[normalized]


def _validate_family(unit: str, expected_family: str) -> None:
    parsed = _parse_unit(unit)
    if parsed.family != expected_family:
        raise ValueError(
            f"unit {unit} belongs to {parsed.family}, expected {expected_family}"
        )


__all__ = [
    "SI_PREFIX_FACTORS",
    "convert_absorbed_dose",
    "convert_activity",
    "convert_dose_equivalent",
    "convert_radiation_unit",
    "prefix_factor",
]
