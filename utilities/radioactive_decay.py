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

alpha_decay(nuclide, nuclide_db)
    Returns the daughter :class:`~utilities.nuclide.Nuclide` after alpha decay.

beta_minus_decay(nuclide, nuclide_db)
    Returns the daughter :class:`~utilities.nuclide.Nuclide` after β⁻ decay.

beta_plus_decay(nuclide, nuclide_db)
    Returns the daughter :class:`~utilities.nuclide.Nuclide` after β⁺ decay.

electron_capture(nuclide, nuclide_db)
    Returns the daughter :class:`~utilities.nuclide.Nuclide` after electron
    capture.

decay_chain(nuclide, nuclide_db)
    Returns the ordered list of :class:`~utilities.nuclide.Nuclide` objects
    from *nuclide* to a stable end-product, following the most probable decay
    mode at each step.

plot_decay_chain(chain, output_path)
    Renders the decay chain returned by :func:`decay_chain` as a vertical
    flowchart and saves it to *output_path*.

Typical usage::

    from utilities.radioactive_decay import activity_at_time, decays_in_period
    from utilities.radioactive_decay import decay_chain, plot_decay_chain
    from utilities.nuclide import load_nuclides

    T_HALF_CO60 = 1.66348e8  # seconds
    A0 = 3.7e10              # 1 Ci in Bq

    # Activity after one year
    A = activity_at_time(A0, T_HALF_CO60, 3.15576e7)

    # Total decays in a 1-hour measurement starting now
    N = decays_in_period(A0, T_HALF_CO60, t_start=0, duration=3600)

    # Full decay chain for Co-60, saved to PNG
    nuclides = load_nuclides()
    chain = decay_chain(nuclides["Co60"], nuclides)
    plot_decay_chain(chain, "co60_chain.png")
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid circular import at runtime; Nuclide is used only for type hints
    from utilities.nuclide import Nuclide


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


# ---------------------------------------------------------------------------
# Decay-mode helpers
# ---------------------------------------------------------------------------

def _lookup_by_ZA(nuclide_db: dict[str, Nuclide], Z: int, A: int) -> Nuclide:
    """Return the :class:`~utilities.nuclide.Nuclide` with the given *Z* and *A*.

    Args:
        nuclide_db: Dictionary mapping nuclide names to :class:`~utilities.nuclide.Nuclide`
            instances, as returned by :func:`~utilities.nuclide.load_nuclides`.
        Z: Atomic number of the target nuclide.
        A: Mass number of the target nuclide.

    Returns:
        The matching :class:`~utilities.nuclide.Nuclide`.

    Raises:
        KeyError: If no nuclide with the requested *Z* and *A* exists in
            *nuclide_db*.
    """
    for nuclide in nuclide_db.values():
        if nuclide.Z == Z and nuclide.A == A:
            return nuclide
    raise KeyError(f"No nuclide with Z={Z}, A={A} found in the database.")


def alpha_decay(nuclide: Nuclide, nuclide_db: dict[str, Nuclide]) -> Nuclide:
    """Return the daughter nuclide after alpha decay.

    Alpha decay emits a ⁴He nucleus, reducing the parent by Z − 2 and A − 4.

    Args:
        nuclide: The parent :class:`~utilities.nuclide.Nuclide`.
        nuclide_db: Dictionary mapping nuclide names to
            :class:`~utilities.nuclide.Nuclide` instances.

    Returns:
        The daughter :class:`~utilities.nuclide.Nuclide`.

    Raises:
        ValueError: If *nuclide* is stable.
        KeyError: If the daughter nuclide is not present in *nuclide_db*.
    """
    if nuclide.stable:
        raise ValueError(f"{nuclide.name!r} is stable and does not undergo alpha decay.")
    return _lookup_by_ZA(nuclide_db, nuclide.Z - 2, nuclide.A - 4)


def beta_minus_decay(nuclide: Nuclide, nuclide_db: dict[str, Nuclide]) -> Nuclide:
    """Return the daughter nuclide after β⁻ decay.

    β⁻ decay converts a neutron to a proton, increasing Z by 1 while A is
    unchanged.

    Args:
        nuclide: The parent :class:`~utilities.nuclide.Nuclide`.
        nuclide_db: Dictionary mapping nuclide names to
            :class:`~utilities.nuclide.Nuclide` instances.

    Returns:
        The daughter :class:`~utilities.nuclide.Nuclide`.

    Raises:
        ValueError: If *nuclide* is stable.
        KeyError: If the daughter nuclide is not present in *nuclide_db*.
    """
    if nuclide.stable:
        raise ValueError(f"{nuclide.name!r} is stable and does not undergo β⁻ decay.")
    return _lookup_by_ZA(nuclide_db, nuclide.Z + 1, nuclide.A)


def beta_plus_decay(nuclide: Nuclide, nuclide_db: dict[str, Nuclide]) -> Nuclide:
    """Return the daughter nuclide after β⁺ decay.

    β⁺ decay converts a proton to a neutron, decreasing Z by 1 while A is
    unchanged.

    Args:
        nuclide: The parent :class:`~utilities.nuclide.Nuclide`.
        nuclide_db: Dictionary mapping nuclide names to
            :class:`~utilities.nuclide.Nuclide` instances.

    Returns:
        The daughter :class:`~utilities.nuclide.Nuclide`.

    Raises:
        ValueError: If *nuclide* is stable.
        KeyError: If the daughter nuclide is not present in *nuclide_db*.
    """
    if nuclide.stable:
        raise ValueError(f"{nuclide.name!r} is stable and does not undergo β⁺ decay.")
    return _lookup_by_ZA(nuclide_db, nuclide.Z - 1, nuclide.A)


def electron_capture(nuclide: Nuclide, nuclide_db: dict[str, Nuclide]) -> Nuclide:
    """Return the daughter nuclide after electron capture (EC).

    Electron capture has the same nuclear effect as β⁺ decay: Z decreases by 1
    while A is unchanged.

    Args:
        nuclide: The parent :class:`~utilities.nuclide.Nuclide`.
        nuclide_db: Dictionary mapping nuclide names to
            :class:`~utilities.nuclide.Nuclide` instances.

    Returns:
        The daughter :class:`~utilities.nuclide.Nuclide`.

    Raises:
        ValueError: If *nuclide* is stable.
        KeyError: If the daughter nuclide is not present in *nuclide_db*.
    """
    if nuclide.stable:
        raise ValueError(
            f"{nuclide.name!r} is stable and does not undergo electron capture."
        )
    return _lookup_by_ZA(nuclide_db, nuclide.Z - 1, nuclide.A)


# ---------------------------------------------------------------------------
# Decay chain
# ---------------------------------------------------------------------------

#: Decay-mode string → (ΔZ, ΔA) nuclear change.
#: Isomeric transition (IT) is intentionally excluded: ΔZ = ΔA = 0 would
#: return the parent itself via Z/A lookup.  IT daughters must be resolved via
#: the ``"daughter"`` key in the decay-mode data.
_DECAY_MODE_DELTAS: dict[str, tuple[int, int]] = {
    "alpha": (-2, -4),
    "beta-": (+1, 0),
    "beta+": (-1, 0),
    "electron_capture": (-1, 0),
    "ec": (-1, 0),
    "proton": (-1, -1),
    "neutron": (0, -1),
}


def decay_chain(
    nuclide: Nuclide,
    nuclide_db: dict[str, Nuclide],
    max_steps: int = 200,
) -> list[Nuclide]:
    """Return the decay chain from *nuclide* to a stable end-product.

    At each step the most probable decay mode (highest branching fraction) is
    followed.  If the :attr:`~utilities.nuclide.Nuclide.decay_modes` list
    carries a ``"daughter"`` key, the named daughter is looked up directly in
    *nuclide_db*; otherwise the daughter is found by applying the standard
    nuclear-change rules (ΔZ, ΔA) for that decay mode.

    The returned list starts with *nuclide* and ends with the stable nuclide
    that terminates the chain.

    Args:
        nuclide: Starting :class:`~utilities.nuclide.Nuclide`.
        nuclide_db: Dictionary mapping nuclide names to
            :class:`~utilities.nuclide.Nuclide` instances.
        max_steps: Maximum number of decay steps to follow before raising an
            error (guards against infinite loops for nuclides not in the
            database).  Defaults to 200.

    Returns:
        Ordered list of :class:`~utilities.nuclide.Nuclide` objects from
        *nuclide* (inclusive) to the stable end-product (inclusive).

    Raises:
        ValueError: If *max_steps* is reached without finding a stable nuclide,
            or if a decay mode is encountered whose nuclear-change rules are
            unknown and no ``"daughter"`` key is present.
        KeyError: If a daughter nuclide referenced by name or inferred by Z/A
            is not present in *nuclide_db*.
    """
    # Build a (Z, A) → Nuclide index once for O(1) lookups during chain traversal.
    za_index: dict[tuple[int, int], Nuclide] = {
        (n.Z, n.A): n for n in nuclide_db.values()
    }

    chain: list[Nuclide] = [nuclide]
    seen: set[str] = {nuclide.name}

    current = nuclide
    for _ in range(max_steps):
        if current.stable:
            break

        # Pick the decay mode with the highest branching fraction.
        dominant = max(current.decay_modes, key=lambda m: m.get("branching_fraction", 0.0))
        mode = dominant["mode"].lower()

        # Resolve daughter: prefer explicit name if present in the data.
        daughter_name: str | None = dominant.get("daughter")
        if daughter_name is not None:
            if daughter_name not in nuclide_db:
                # Named daughter is not in the database – it is either stable
                # or outside the scope of the loaded data.  The chain ends here.
                break
            daughter = nuclide_db[daughter_name]
        else:
            if mode not in _DECAY_MODE_DELTAS:
                raise ValueError(
                    f"Unknown decay mode {dominant['mode']!r} for {current.name!r}; "
                    "cannot determine daughter nuclide."
                )
            dZ, dA = _DECAY_MODE_DELTAS[mode]
            key = (current.Z + dZ, current.A + dA)
            if key not in za_index:
                raise KeyError(
                    f"No nuclide with Z={key[0]}, A={key[1]} found in the database."
                )
            daughter = za_index[key]

        if daughter.name in seen:
            # Cycle detected (should not happen in physical data, but guard anyway).
            break

        chain.append(daughter)
        seen.add(daughter.name)
        current = daughter
    else:
        raise ValueError(
            f"Decay chain starting from {nuclide.name!r} did not reach a stable "
            f"nuclide within {max_steps} steps."
        )

    return chain


# ---------------------------------------------------------------------------
# Decay chain visualisation
# ---------------------------------------------------------------------------

#: Colour used for each decay mode in the flowchart arrows.
_MODE_COLOURS: dict[str, str] = {
    "alpha": "#e05c00",
    "beta-": "#1565c0",
    "beta+": "#6a1fa0",
    "electron_capture": "#6a1fa0",
    "ec": "#6a1fa0",
    "isomeric_transition": "#2e7d32",
    "it": "#2e7d32",
    "proton": "#c62828",
    "neutron": "#37474f",
}
_DEFAULT_ARROW_COLOUR = "#555555"

#: Human-readable labels for the mode codes used in decay_modes data.
_MODE_LABELS: dict[str, str] = {
    "alpha": "α",
    "beta-": "β⁻",
    "beta+": "β⁺",
    "electron_capture": "ε",
    "ec": "ε",
    "isomeric_transition": "IT",
    "it": "IT",
    "proton": "p",
    "neutron": "n",
}


def plot_decay_chain(
    chain: list[Nuclide],
    output_path: str | Path,
    dpi: int = 150,
) -> Path:
    """Render *chain* as a vertical flowchart and save to *output_path*.

    Each nuclide in the chain is drawn as a labelled box.  Arrows between
    consecutive boxes are annotated with the dominant decay mode (from the
    parent's ``decay_modes`` list) and the parent's half-life.  Stable
    end-products are highlighted with a green border.

    The output format is inferred from the file extension (e.g. ``.png``,
    ``.pdf``, ``.svg``).

    Args:
        chain: Ordered list of :class:`~utilities.nuclide.Nuclide` objects as
            returned by :func:`decay_chain`.  Must contain at least one entry.
        output_path: Destination file path.  Parent directories must already
            exist.
        dpi: Resolution for raster formats (PNG/JPEG).  Ignored for vector
            formats.  Defaults to 150.

    Returns:
        The resolved :class:`~pathlib.Path` to the saved file.

    Raises:
        ValueError: If *chain* is empty.
        ImportError: If ``matplotlib`` is not installed.
    """
    if not chain:
        raise ValueError("chain must contain at least one nuclide.")

    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "matplotlib is required for plot_decay_chain. "
            "Install it with: pip install matplotlib"
        ) from exc

    n_nodes = len(chain)
    fig_height = max(3.0, n_nodes * 1.4 + 0.8)
    fig, ax = plt.subplots(figsize=(4.0, fig_height))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, n_nodes - 0.5)
    ax.axis("off")

    box_w = 0.55
    box_h = 0.55
    box_x = 0.5 - box_w / 2  # centred horizontally

    # y-coordinates: top nuclide at the top of the figure
    y_centres = [n_nodes - 1 - i for i in range(n_nodes)]

    for i, nuclide in enumerate(chain):
        yc = y_centres[i]
        y0 = yc - box_h / 2

        edge_colour = "#2e7d32" if nuclide.stable else "#333333"
        linewidth = 2.0 if nuclide.stable else 1.2

        rect = mpatches.FancyBboxPatch(
            (box_x, y0),
            box_w,
            box_h,
            boxstyle="round,pad=0.02",
            linewidth=linewidth,
            edgecolor=edge_colour,
            facecolor="#f5f5f5",
        )
        ax.add_patch(rect)

        # Nuclide name (bold)
        ax.text(
            0.5, yc + 0.06,
            nuclide.long_name,
            ha="center", va="center",
            fontsize=9, fontweight="bold",
            color="#111111",
        )

        # Half-life or "stable"
        if nuclide.stable:
            hl_text = "stable"
        elif nuclide.half_life_years is not None:
            hl = nuclide.half_life_years
            if hl >= 1.0:
                hl_text = f"T½ = {hl:.4g} y"
            else:
                hl_text = f"T½ = {nuclide.half_life_seconds:.4g} s"
        else:
            hl_text = ""

        ax.text(
            0.5, yc - 0.12,
            hl_text,
            ha="center", va="center",
            fontsize=7, color="#555555",
        )

        # Draw arrow + decay mode label between this box and the next
        if i < n_nodes - 1:
            # arrow_start_y: bottom edge of the current box (arrow origin)
            # arrow_end_y: top edge of the next (lower) box (arrowhead destination)
            arrow_start_y = y0
            arrow_end_y = y_centres[i + 1] + box_h / 2 + 0.02

            # Determine dominant decay mode for annotation
            mode_str = ""
            arrow_colour = _DEFAULT_ARROW_COLOUR
            if nuclide.decay_modes:
                dominant = max(
                    nuclide.decay_modes,
                    key=lambda m: m.get("branching_fraction", 0.0),
                )
                raw_mode = dominant["mode"].lower()
                mode_str = _MODE_LABELS.get(raw_mode, dominant["mode"])
                bf = dominant.get("branching_fraction")
                if bf is not None and bf < 1.0:
                    mode_str += f" ({bf * 100:.1f}%)"
                arrow_colour = _MODE_COLOURS.get(raw_mode, _DEFAULT_ARROW_COLOUR)

            ax.annotate(
                "",
                xy=(0.5, arrow_end_y),
                xytext=(0.5, arrow_start_y),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=arrow_colour,
                    lw=1.5,
                ),
            )
            mid_y = (arrow_start_y + arrow_end_y) / 2
            ax.text(
                0.5 + box_w / 2 + 0.04, mid_y,
                mode_str,
                ha="left", va="center",
                fontsize=8, color=arrow_colour,
            )

    ax.set_title(
        f"Decay chain: {chain[0].long_name}",
        fontsize=10, pad=8,
    )

    out = Path(output_path)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out
