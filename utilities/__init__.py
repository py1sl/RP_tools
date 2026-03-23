"""RP_tools utilities package.

Shared data-handling classes and common calculation functions used across all
RP_tools tool packages. Individual tools (e.g. Gaussian plume model, skin dose
model) each live in their own top-level folder and import from this package.
"""

from utilities.nuclide import Nuclide, load_nuclides
from utilities.icrp_data import ICRPDataLibrary, ICRPTable, load_icrp_data
from utilities.immersion_dose import (
    ImmersionDoseCalculator,
    immersion_dose_rate_from_concentration,
    immersion_dose_rate_on_grid,
)
from utilities.ground_plane_dose import (
    SemiInfinitePlaneDoseCalculator,
    semi_infinite_plane_dose_rate_from_deposition,
    semi_infinite_plane_dose_rate_on_grid,
)
from utilities.radioactive_decay import (
    activity_at_time,
    decay_constant,
    decays_in_period,
    time_to_activity,
)

__all__ = [
    "Nuclide",
    "load_nuclides",
    "ICRPTable",
    "ICRPDataLibrary",
    "load_icrp_data",
    "ImmersionDoseCalculator",
    "immersion_dose_rate_from_concentration",
    "immersion_dose_rate_on_grid",
    "SemiInfinitePlaneDoseCalculator",
    "semi_infinite_plane_dose_rate_from_deposition",
    "semi_infinite_plane_dose_rate_on_grid",
    "activity_at_time",
    "decay_constant",
    "decays_in_period",
    "time_to_activity",
]
