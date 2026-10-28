"""Compartment (biokinetic) models for RP_tools.

A compartment model is a set of linear first-order transfers between
compartments, plus optional loss (excretion) and radioactive decay:

    dq/dt = K q,   K[i, i] = -(sum of outgoing rates + decay constant),
                   K[j, i] = transfer rate from compartment i to j.

The solution q(t) = expm(K t) q0 is computed exactly.  Rates are in 1/time
and times in the same unit (days are typical for biokinetics).

Classes / functions
-------------------
CompartmentModel
    General linear compartment model.
single_compartment_retention(half_life_biological, t, ...)
    Retention in one organ with biological clearance (and optional decay).
effective_half_life(half_life_physical, half_life_biological)
    Combined effective half-life.
iodine_thyroid_model(...)
    Simple blood -> thyroid -> excreta model for radioiodine.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from utilities.radioactive_decay import decay_constant


def _expm(matrix: np.ndarray) -> np.ndarray:
    """Matrix exponential via scaling-and-squaring with a Taylor series."""
    norm = np.linalg.norm(matrix, np.inf)
    squarings = max(0, int(math.ceil(math.log2(norm))) + 1) if norm > 0.5 else 0
    a = matrix / (2.0 ** squarings)
    result = np.eye(matrix.shape[0])
    term = np.eye(matrix.shape[0])
    for k in range(1, 30):
        term = term @ a / k
        result = result + term
    for _ in range(squarings):
        result = result @ result
    return result


def effective_half_life(half_life_physical: float, half_life_biological: float) -> float:
    """Return T_eff = Tp * Tb / (Tp + Tb). Both must be positive."""
    if half_life_physical <= 0 or half_life_biological <= 0:
        raise ValueError("half-lives must be positive")
    return half_life_physical * half_life_biological / (
        half_life_physical + half_life_biological
    )


class CompartmentModel:
    """Linear compartment model with transfers, excretion and decay.

    Args:
        compartments: Ordered compartment names (e.g. ``["blood", "thyroid"]``).
        transfers: Mapping ``(source, target) -> rate`` (1/time).  The target
            may be a name not in *compartments* (e.g. ``"excreta"``) in which
            case it is treated as a sink and not tracked.
        half_life: Optional radioactive half-life (same time unit) applied to
            every compartment.
    """

    def __init__(
        self,
        compartments: List[str],
        transfers: Dict[Tuple[str, str], float],
        half_life: Optional[float] = None,
    ) -> None:
        if len(set(compartments)) != len(compartments) or not compartments:
            raise ValueError("compartments must be non-empty and unique")
        self.compartments = list(compartments)
        self._index = {c: i for i, c in enumerate(self.compartments)}
        n = len(self.compartments)
        k = np.zeros((n, n))
        for (src, dst), rate in transfers.items():
            if src not in self._index:
                raise ValueError(f"unknown source compartment: {src!r}")
            if rate < 0:
                raise ValueError("transfer rates must be non-negative")
            k[self._index[src], self._index[src]] -= rate
            if dst in self._index:
                k[self._index[dst], self._index[src]] += rate
        if half_life is not None:
            lam = decay_constant(half_life)
            k -= lam * np.eye(n)
        self.matrix = k

    def _initial(self, initial: Dict[str, float]) -> np.ndarray:
        q0 = np.zeros(len(self.compartments))
        for name, value in initial.items():
            if name not in self._index:
                raise ValueError(f"unknown compartment: {name!r}")
            q0[self._index[name]] = value
        return q0

    def amounts(self, initial: Dict[str, float], t: float) -> Dict[str, float]:
        """Return the amount in each compartment at time *t* >= 0."""
        if t < 0:
            raise ValueError("t must be non-negative")
        q = _expm(self.matrix * t) @ self._initial(initial)
        return dict(zip(self.compartments, q.tolist()))

    def time_series(
        self, initial: Dict[str, float], times: List[float]
    ) -> Dict[str, List[float]]:
        """Return amounts in each compartment at each of *times*."""
        out: Dict[str, List[float]] = {c: [] for c in self.compartments}
        for t in times:
            for name, value in self.amounts(initial, t).items():
                out[name].append(value)
        return out

    def total_decays(self, initial: Dict[str, float], compartment: str, t: float) -> float:
        """Integral of the amount in *compartment* over [0, t].

        Only meaningful for radioactive models; multiply by the decay
        constant to obtain the number of disintegrations.
        """
        if t < 0:
            raise ValueError("t must be non-negative")
        n = len(self.compartments)
        # Augmented system: d/dt [q; I] = [[K, 0], [I_n, 0]] [q; I]
        aug = np.zeros((2 * n, 2 * n))
        aug[:n, :n] = self.matrix
        aug[n:, :n] = np.eye(n)
        state = _expm(aug * t) @ np.concatenate([self._initial(initial), np.zeros(n)])
        return float(state[n + self._index[compartment]])


def single_compartment_retention(
    half_life_biological: float, t: float, half_life_physical: Optional[float] = None
) -> float:
    """Fraction retained in one compartment at time *t*.

    Uses the biological half-life, combined with the physical half-life when
    given.
    """
    if t < 0:
        raise ValueError("t must be non-negative")
    half_life = (
        half_life_biological
        if half_life_physical is None
        else effective_half_life(half_life_physical, half_life_biological)
    )
    return math.exp(-decay_constant(half_life) * t)


def iodine_thyroid_model(
    half_life_physical: Optional[float] = 8.02,
    thyroid_uptake_fraction: float = 0.3,
    thyroid_biological_half_life: float = 80.0,
    blood_half_life: float = 0.25,
) -> CompartmentModel:
    """Simple radioiodine model (times in days; defaults ~ I-131, adult).

    Iodine leaves blood with *blood_half_life*; a fraction
    *thyroid_uptake_fraction* goes to the thyroid, the rest to excreta.  The
    thyroid clears with *thyroid_biological_half_life* (ICRP default 80 d).
    Pass ``half_life_physical=None`` to ignore radioactive decay.
    """
    if not 0 <= thyroid_uptake_fraction <= 1:
        raise ValueError("thyroid_uptake_fraction must be in [0, 1]")
    k_blood = decay_constant(blood_half_life)
    k_thy = decay_constant(thyroid_biological_half_life)
    transfers = {
        ("blood", "thyroid"): k_blood * thyroid_uptake_fraction,
        ("blood", "excreta"): k_blood * (1 - thyroid_uptake_fraction),
        ("thyroid", "excreta"): k_thy,
    }
    return CompartmentModel(["blood", "thyroid"], transfers, half_life_physical)
