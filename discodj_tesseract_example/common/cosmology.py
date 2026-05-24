"""Shared cosmological defaults for the DiscoDJ examples."""
from __future__ import annotations
from dataclasses import dataclass

DEFAULT_COSMOLOGY = dict(
    Omega_c=0.259605,
    Omega_b=0.0488911,
    h=0.67742,
    n_s=0.96822,
    sigma8=0.808992,
)


@dataclass(frozen=True)
class ModelDefaults:
    n_steps: int = 10
    boxsize: float = 100.0
    a_ini: float = 0.02
    a_end: float = 1.0
    lpt_order: int = 1
    mass_assignment_order: int = 3


DEFAULT_MODEL = ModelDefaults()
