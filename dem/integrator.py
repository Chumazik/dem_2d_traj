"""Шаг Velocity Verlet.

Использует Numba-JIT ядро из :mod:`dem.jit_kernels` если доступно
и ``config.use_jit=True``. Иначе работает в чистом Python.
"""

import numpy as np

from .force_calculation import compute_all_forces
from .jit_kernels import _velocity_verlet_step


def _pack_state(particles):
    """Распаковывает состояние частиц в SoA-массивы."""
    n = len(particles)
    pos = np.empty((n, 2), dtype=np.float64)
    vel = np.empty((n, 2), dtype=np.float64)
    ang_vel = np.empty(n, dtype=np.float64)
    force = np.empty((n, 2), dtype=np.float64)
    torque = np.empty(n, dtype=np.float64)
    mass = np.empty(n, dtype=np.float64)
    inertia = np.empty(n, dtype=np.float64)
    for i, p in enumerate(particles):
        pos[i, 0] = p.pos[0]
        pos[i, 1] = p.pos[1]
        vel[i, 0] = p.vel[0]
        vel[i, 1] = p.vel[1]
        ang_vel[i] = p.ang_vel
        force[i, 0] = p.force[0]
        force[i, 1] = p.force[1]
        torque[i] = p.torque
        mass[i] = p.mass
        inertia[i] = p.inertia
    return pos, vel, ang_vel, force, torque, mass, inertia


def _unpack_state(particles, pos, vel, ang_vel):
    """Записывает обновлённые pos/vel/ang_vel обратно в Particle."""
    for i, p in enumerate(particles):
        p.pos[0] = pos[i, 0]
        p.pos[1] = pos[i, 1]
        p.vel[0] = vel[i, 0]
        p.vel[1] = vel[i, 1]
        p.ang_vel = ang_vel[i]


def velocity_verlet_step(particles, dt, contact_model, boundaries):
    """Один шаг интегратора Velocity Verlet.

    Если доступен Numba и ``config.use_jit=True`` (по умолчанию), ядро
    :func:`dem.jit_kernels._velocity_verlet_step` выполняет шаг без
    Python-overhead.
    """
    cfg = getattr(contact_model, "config", None)
    use_jit = bool(getattr(cfg, "use_jit", True)) if cfg is not None else True

    if use_jit:
        try:
            pos, vel, ang_vel, force, torque, mass, inertia = _pack_state(particles)
            _velocity_verlet_step(pos, vel, ang_vel, force, torque, mass, inertia, dt)
            _unpack_state(particles, pos, vel, ang_vel)
            # Силы уже использованы; обнуляем после записи новых
            for p in particles:
                p.reset_force()
            compute_all_forces(particles, boundaries, contact_model)
            return
        except Exception:
            # На любой ошибке JIT-пути (например, нет Numba) — fallback
            pass

    # ----- Python-путь -----
    for p in particles:
        a = p.force / p.mass
        alpha = p.torque / p.inertia
        p.vel += 0.5 * a * dt
        p.ang_vel += 0.5 * alpha * dt

    for p in particles:
        p.pos += p.vel * dt

    for p in particles:
        p.reset_force()

    compute_all_forces(particles, boundaries, contact_model)

    for p in particles:
        a = p.force / p.mass
        alpha = p.torque / p.inertia
        p.vel += 0.5 * a * dt
        p.ang_vel += 0.5 * alpha * dt

    for p in particles:
        p.update_history()
