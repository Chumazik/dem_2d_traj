"""Шаг Velocity Verlet.

Использует Numba-JIT ядро из :mod:`dem.jit_kernels` если доступно
и ``config.use_jit=True``. Если ``config.use_gpu=True`` и CuPy
доступен, шаг делается на GPU. Иначе — чистый Python (полный
двух‑шаговый Verlet + ``update_history``).
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

    Выбор бэкенда:
    1. ``config.use_gpu=True`` + доступный CuPy →
       :func:`dem.gpu_backend.velocity_verlet_step_cupy` (полушаг + позиции);
    2. ``config.use_jit=True`` (по умолчанию) + Numba →
       :func:`dem.jit_kernels._velocity_verlet_step`;
    3. иначе → чистый Python (полный 2‑шаговый Verlet + history).

    Во всех ветках делается полный двухшаговый Verlet + ``update_history``
    для численной согласованности.
    """
    cfg = getattr(contact_model, "config", None)
    use_jit = bool(getattr(cfg, "use_jit", True)) if cfg is not None else True
    use_gpu = bool(getattr(cfg, "use_gpu", False)) if cfg is not None else False

    if use_gpu:
        try:
            from . import gpu_backend
            if gpu_backend.is_available():
                # Первый полушаг (vel) + позиции делает GPU.
                gpu_backend.velocity_verlet_step_cupy(particles, dt)
                # Пересчёт сил (Numba/GPU/CPU для пар + CPU для границ) и второй полушаг.
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
                return
        except Exception:
            # мягкий фолбэк на Numba
            pass

    if use_jit:
        try:
            pos, vel, ang_vel, force, torque, mass, inertia = _pack_state(particles)
            _velocity_verlet_step(pos, vel, ang_vel, force, torque, mass, inertia, dt)
            _unpack_state(particles, pos, vel, ang_vel)
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
            return
        except Exception:
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
