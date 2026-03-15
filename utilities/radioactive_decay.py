"""Radioactive decay calculation functions for RP_tools.

This module is part of the ``utilities`` package, which provides shared
data-handling classes and common functions used across all RP_tools tool
packages (Gaussian plume model, skin dose, ingestion dose, etc.).

All functions operate on scalar values. Times and half-lives must be provided
in **consistent units** (seconds are recommended). Activities are in Bq
(decays per second) when seconds are used, or in whatever activity unit is
consistent with the supplied time unit.

Functions
---------
decay_constant(half_life)
    Returns the decay constant λ.

activity_at_time(A0, half_life, t)
    Returns the activity at time *t* after an initial activity *A0*.

decays_in_period(A0, half_life, t_start, duration)
    Returns the total number of decays that occur in the interval
    [t_start, t_start + duration].

time_to_activity(A0, A_target, half_life)
    Returns the time required to decay from *A0* to *A_target*.

Typical usage::

    from utilities.radioactive_decay import activity_at_time, decays_in_period

    T_HALF_CO60 = 1.66348e8  # seconds
    A0 = 3.7e10              # 1 Ci in Bq

    # Activity after one year
    A = activity_at_time(A0, T_HALF_CO60, 3.15576e7)

    # Total decays in a 1-hour measurement starting now
    N = decays_in_period(A0, T_HALF_CO60, t_start=0, duration=3600)
"""

from __future__ import annotations

import math


def decay_constant(half_life: float) -> float:
    """Return the decay constant λ = ln(2) / T½.

    Args:
        half_life: Half-life in any consistent time unit. Must be positive.

    Returns:
        Decay constant λ in units of 1 / (time unit of *half_life*).

    Raises:
        ValueError: If *half_life* is not positive.
    """
    if half_life <= 0:
        raise ValueError(f"half_life must be positive, got {half_life}")
    return math.log(2) / half_life


def activity_at_time(A0: float, half_life: float, t: float) -> float:
    """Return the activity at time *t* after an initial activity *A0*.

    Uses the standard exponential decay law:
    ``A(t) = A0 * exp(-λ * t)``

    Args:
        A0: Initial activity (Bq or any consistent activity unit). Must be
            non-negative.
        half_life: Half-life in the same time unit as *t*. Must be positive.
        t: Time elapsed since the activity was *A0*. Must be non-negative.

    Returns:
        Activity at time *t* in the same unit as *A0*.

    Raises:
        ValueError: If *A0* is negative, *half_life* is not positive, or *t*
            is negative.
    """
    if A0 < 0:
        raise ValueError(f"Initial activity A0 must be non-negative, got {A0}")
    if t < 0:
        raise ValueError(f"Time t must be non-negative, got {t}")

    lam = decay_constant(half_life)
    return A0 * math.exp(-lam * t)


def decays_in_period(
    A0: float,
    half_life: float,
    t_start: float,
    duration: float,
) -> float:
    """Return the total number of decays in the interval [t_start, t_start + duration].

    Integrates the activity over the interval:
    ``N = ∫_{t_start}^{t_start+duration} A0 * exp(-λ*t) dt``
      ``= (A0 / λ) * exp(-λ*t_start) * (1 − exp(-λ*duration))``

    Args:
        A0: Activity at time *t* = 0 (Bq or any consistent unit). Must be
            non-negative.
        half_life: Half-life in the same time unit as *t_start* and *duration*.
            Must be positive.
        t_start: Start of the time interval. Must be non-negative.
        duration: Length of the interval. Must be non-negative.

    Returns:
        Total number of decays (dimensionless count) in the interval.

    Raises:
        ValueError: If any argument violates its constraints.
    """
    if A0 < 0:
        raise ValueError(f"Initial activity A0 must be non-negative, got {A0}")
    if t_start < 0:
        raise ValueError(f"t_start must be non-negative, got {t_start}")
    if duration < 0:
        raise ValueError(f"duration must be non-negative, got {duration}")

    if duration == 0:
        return 0.0

    lam = decay_constant(half_life)
    # Activity at the start of the interval
    A_start = A0 * math.exp(-lam * t_start)
    # Integrate: (A_start / λ) * (1 − exp(−λ * duration))
    return (A_start / lam) * (1.0 - math.exp(-lam * duration))


def time_to_activity(A0: float, A_target: float, half_life: float) -> float:
    """Return the time required to decay from *A0* to *A_target*.

    Derived from ``A_target = A0 * exp(-λ * t)``:
    ``t = ln(A0 / A_target) / λ``

    Args:
        A0: Initial activity. Must be positive.
        A_target: Target activity. Must be positive and less than or equal to
            *A0*.
        half_life: Half-life in any consistent time unit. Must be positive.

    Returns:
        Time to reach *A_target* in the same unit as *half_life*.

    Raises:
        ValueError: If *A0* or *A_target* are not positive, if *A_target*
            exceeds *A0*, or if *half_life* is not positive.
    """
    if A0 <= 0:
        raise ValueError(f"Initial activity A0 must be positive, got {A0}")
    if A_target <= 0:
        raise ValueError(f"Target activity A_target must be positive, got {A_target}")
    if A_target > A0:
        raise ValueError(
            f"A_target ({A_target}) cannot exceed A0 ({A0}); "
            "activity cannot increase by radioactive decay alone."
        )

    lam = decay_constant(half_life)
    return math.log(A0 / A_target) / lam
