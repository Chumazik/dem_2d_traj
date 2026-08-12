"""GPU-бэкенд для горячего цикла DEM (CuPy).

Предоставляет батчевые реализации двух расчётных ядер:

* :func:`compute_pairwise_forces_cupy` — парные контакты частица-частица,
  аналог :func:`dem.jit_kernels.pairwise._pairwise_particle_forces`;
* :func:`velocity_verlet_step_cupy` — шаг Velocity Verlet (полушаг
  скоростей + шаг позиций) на GPU, аналог
  :func:`dem.jit_kernels.integrator._velocity_verlet_step`.

Если CuPy недоступен или CUDA-устройств нет, :func:`is_available`
возвращает ``False``. Вызывающая сторона (``dem.force_calculation``)
обрабатывает это и прозрачно переключается на путь Numba или CPU.
"""

from __future__ import annotations

import math

import numpy as np

try:  # pragma: no cover - exercised via tests with mocked CuPy
    import cupy as _cp
    _CUPY_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # ImportError / OSError
    _cp = None  # type: ignore[assignment]
    _CUPY_IMPORT_ERROR = exc


# ---------------------------------------------------------------------------
# Доступность бэкенда
# ---------------------------------------------------------------------------
def is_available() -> bool:
    """``True``, если CuPy установлен и доступно хотя бы одно CUDA-устройство."""
    if _cp is None:
        return False
    try:
        return int(_cp.cuda.runtime.getDeviceCount()) > 0
    except Exception:
        return False


def import_error() -> Exception | None:
    """Возвращает исключение при импорте CuPy (для диагностики)."""
    return _CUPY_IMPORT_ERROR


# ---------------------------------------------------------------------------
# Парные контактные силы
# ---------------------------------------------------------------------------
def compute_pairwise_forces_cupy(particles, contact_model) -> None:
    """Батчевая парная нормальная+касательная сила + момент качения на GPU.

    Результат добавляется в ``particle.force`` / ``particle.torque`` каждой
    частицы (так же, как и в чисто-Python пути). Перед вызовом поля
    ``force``/``torque`` частиц должны быть обнулены вызывающей стороной.
    """
    if not _cp or not is_available():
        raise RuntimeError("CuPy GPU backend unavailable")

    n = len(particles)
    if n == 0:
        return

    cp = _cp

    pos = cp.empty((n, 2), dtype=cp.float64)
    vel = cp.empty((n, 2), dtype=cp.float64)
    ang_vel = cp.empty(n, dtype=cp.float64)
    radius = cp.empty(n, dtype=cp.float64)
    for i, p in enumerate(particles):
        pos[i, 0] = float(p.pos[0])
        pos[i, 1] = float(p.pos[1])
        vel[i, 0] = float(p.vel[0])
        vel[i, 1] = float(p.vel[1])
        ang_vel[i] = float(p.ang_vel)
        radius[i] = float(p.radius)

    # Попарные смещения
    dx = pos[:, 0][:, None] - pos[:, 0][None, :]
    dy = pos[:, 1][:, None] - pos[:, 1][None, :]
    rsum = radius[:, None] + radius[None, :]
    dist2 = dx * dx + dy * dy
    dist = cp.sqrt(cp.where(dist2 > 0.0, dist2, 1.0))  # защита от sqrt(0)
    inv_dist = cp.where(dist2 > 0.0, 1.0 / cp.where(dist2 > 0.0, dist2, 1.0), 0.0)

    overlap = rsum - dist
    mask = (overlap > 0.0) & (dist2 > 0.0) & (cp.arange(n)[:, None] < cp.arange(n)[None, :])
    # mask[j,i] = True означает что j>i и есть контакт

    nx = cp.where(mask, dx * inv_dist, 0.0)
    ny = cp.where(mask, dy * inv_dist, 0.0)

    rel_vx = vel[:, 0][:, None] - vel[:, 0][None, :]
    rel_vy = vel[:, 1][:, None] - vel[:, 1][None, :]
    overlap_rate = rel_vx * nx + rel_vy * ny

    kn = float(contact_model.kn)
    e = float(contact_model.restitution_coeff)
    gamma_n = -2.0 * math.sqrt(kn * e)

    fn_scalar = kn * overlap + gamma_n * overlap_rate
    # Только для активных контактов; обнуляем неактивные ячейки
    fn_scalar = cp.where(mask, fn_scalar, 0.0)

    # Силы: на i действует -fn_x, на j — +fn_x (симметрия). Поскольку маска bx<by (i<j),
    # то для частицы с меньшим i: берем сумму по j>i со знаком + для f, на j<i со знаком -.
    # Проще: для каждого i просуммировать с разными знаками для i<j и i>j.
    # Тут: fn[j,i] — это пара выше диагонали, для i<j → +fn на j => суммируем
    # по i (i меньше) -> -fn, по j (j больше) -> +fn. Симметричная матрица mat,
    # и сумма по строкам и столбцам.
    force_x = -fn_scalar * nx  # для частицы в строке i (где i<j) сила -fn
    # Расширить симметрично:
    force_x_full = force_x - force_x.T  # (i,j): (j>i) -fn, (i>j) +fn
    force_y_full = (-fn_scalar * ny) - (-fn_scalar * ny).T

    # Rolling friction (sideways)
    mu_r = float(contact_model.rolling_friction_coeff)
    if mu_r != 0.0 and n >= 2:
        omega_rel = ang_vel[:, None] - ang_vel[None, :]
        r_eff = (
            radius[:, None] * radius[None, :]
            / cp.where((radius[:, None] + radius[None, :]) > 0,
                       radius[:, None] + radius[None, :], 1.0)
        )
        sign_om = cp.sign(omega_rel)
        roll_torque = -mu_r * cp.abs(fn_scalar) * r_eff * sign_om
        roll_torque = cp.where(mask, roll_torque, 0.0)
        # Для пары (i,j) с i<j: на i действует +roll_torque[i,j], на j — -roll_torque[i,j]
        torque_full = roll_torque - roll_torque.T
    else:
        torque_full = cp.zeros((n, n), dtype=cp.float64)

    force_x_sum = force_x_full.sum(axis=1)
    force_y_sum = force_y_full.sum(axis=1)
    torque_sum = torque_full.sum(axis=1)

    f_x_np = cp.asnumpy(force_x_sum)
    f_y_np = cp.asnumpy(force_y_sum)
    t_np = cp.asnumpy(torque_sum)

    for i, p in enumerate(particles):
        p.force[0] += float(f_x_np[i])
        p.force[1] += float(f_y_np[i])
        p.torque += float(t_np[i])


# ---------------------------------------------------------------------------
# Velocity Verlet (один шаг)
# ---------------------------------------------------------------------------
def velocity_verlet_step_cupy(particles, dt) -> None:
    """Шаг Velocity Verlet (полушаг скоростей + шаг позиций) на GPU.

    Идентично ``dem.jit_kernels.integrator._velocity_verlet_step``: читает
    ``particle.force/torque``, модифицирует ``vel/ang_vel/pos``. Поля
    ``force/torque`` сбрасываются вызывающей стороной.
    """
    if not _cp or not is_available():
        raise RuntimeError("CuPy GPU backend unavailable")

    n = len(particles)
    if n == 0:
        return

    cp = _cp

    pos = cp.empty((n, 2), dtype=cp.float64)
    vel = cp.empty((n, 2), dtype=cp.float64)
    ang_vel = cp.empty(n, dtype=cp.float64)
    force = cp.empty((n, 2), dtype=cp.float64)
    torque = cp.empty(n, dtype=cp.float64)
    mass = cp.empty(n, dtype=cp.float64)
    inertia = cp.empty(n, dtype=cp.float64)
    for i, p in enumerate(particles):
        pos[i, 0] = float(p.pos[0])
        pos[i, 1] = float(p.pos[1])
        vel[i, 0] = float(p.vel[0])
        vel[i, 1] = float(p.vel[1])
        ang_vel[i] = float(p.ang_vel)
        force[i, 0] = float(p.force[0])
        force[i, 1] = float(p.force[1])
        torque[i] = float(p.torque)
        mass[i] = float(p.mass)
        inertia[i] = float(p.inertia)

    inv_m = 1.0 / mass
    inv_I = 1.0 / inertia
    ax = force[:, 0] * inv_m
    ay = force[:, 1] * inv_m
    alpha = torque * inv_I
    vel[:, 0] += 0.5 * ax * dt
    vel[:, 1] += 0.5 * ay * dt
    ang_vel += 0.5 * alpha * dt
    pos[:, 0] += vel[:, 0] * dt
    pos[:, 1] += vel[:, 1] * dt

    pos_np = cp.asnumpy(pos)
    vel_np = cp.asnumpy(vel)
    ang_np = cp.asnumpy(ang_vel)
    for i, p in enumerate(particles):
        p.pos[0] = float(pos_np[i, 0])
        p.pos[1] = float(pos_np[i, 1])
        p.vel[0] = float(vel_np[i, 0])
        p.vel[1] = float(vel_np[i, 1])
        p.ang_vel = float(ang_np[i])
