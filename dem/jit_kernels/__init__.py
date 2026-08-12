"""Пакет JIT-ядер для горячего цикла DEM.

Содержит низкоуровневые Numba-ядра:
    * :mod:`dem.jit_kernels.pairwise` – батчевый расчёт сил частица‑частица;
    * :mod:`dem.jit_kernels.wall`     – заготовка для батчевого расчёта
      сил частица‑стенка;
    * :mod:`dem.jit_kernels.integrator` – один шаг Velocity Verlet.
"""

from .pairwise import _pairwise_particle_forces
from .wall import _wall_particle_forces
from .integrator import _velocity_verlet_step

__all__ = [
    "_pairwise_particle_forces",
    "_wall_particle_forces",
    "_velocity_verlet_step",
]
