"""Numba-ядро для одного шага интегратора Velocity Verlet."""

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def _velocity_verlet_step(
    pos,         # (N, 2)
    vel,         # (N, 2)
    ang_vel,     # (N,)
    force,       # (N, 2)
    torque,      # (N,)
    mass,        # (N,)
    inertia,     # (N,)
    dt,
):
    """Один шаг Velocity Verlet без Python-циклов по частицам."""
    N = pos.shape[0]

    # ----- Полушаг скоростей -----
    for i in range(N):
        inv_m = 1.0 / mass[i]
        inv_I = 1.0 / inertia[i]
        ax = force[i, 0] * inv_m
        ay = force[i, 1] * inv_m
        alpha = torque[i] * inv_I
        vel[i, 0] += 0.5 * ax * dt
        vel[i, 1] += 0.5 * ay * dt
        ang_vel[i] += 0.5 * alpha * dt

    # ----- Обновление позиций -----
    for i in range(N):
        pos[i, 0] += vel[i, 0] * dt
        pos[i, 1] += vel[i, 1] * dt
