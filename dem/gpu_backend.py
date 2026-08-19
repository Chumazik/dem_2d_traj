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
def compute_pairwise_forces_cupy(particles, contact_model, tangential_disp=None) -> None:
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
    mass = cp.empty(n, dtype=cp.float64)
    for i, p in enumerate(particles):
        pos[i, 0] = float(p.pos[0])
        pos[i, 1] = float(p.pos[1])
        vel[i, 0] = float(p.vel[0])
        vel[i, 1] = float(p.vel[1])
        ang_vel[i] = float(p.ang_vel)
        radius[i] = float(p.radius)
        mass[i] = float(p.mass)

    # Попарные смещения
    dx = pos[:, 0][:, None] - pos[:, 0][None, :]
    dy = pos[:, 1][:, None] - pos[:, 1][None, :]
    rsum = radius[:, None] + radius[None, :]
    dist2 = dx * dx + dy * dy
    dist = cp.sqrt(cp.where(dist2 > 0.0, dist2, 1.0))  # защита от sqrt(0)
    inv_dist = cp.where(dist2 > 0.0, 1.0 / cp.where(dist2 > 0.0, dist2, 1.0), 0.0)

    overlap = rsum - dist
    mask = (overlap > 0.0) & (dist2 > 0.0) & (cp.arange(n)[:, None] < cp.arange(n)[None, :])
    # mask[i,j] = True означает что i<j и есть контакт

    # Единичный нормальный вектор (i->j), как в CPU-ядре (delta/dist)
    nx = cp.where(mask, dx * inv_dist, 0.0)
    ny = cp.where(mask, dy * inv_dist, 0.0)

    # Относительная скорость j - i
    rel_vx = vel[:, 0][None, :] - vel[:, 0][:, None]
    rel_vy = vel[:, 1][None, :] - vel[:, 1][:, None]
    overlap_rate = rel_vx * nx + rel_vy * ny  # матрица (i,j)

    # Касательный вектор t = (-ny, nx); касательная скорость v_t = rel·t
    tvx = rel_vx * (-ny) + rel_vy * nx

    kn = float(contact_model.kn)
    kt = float(contact_model.kt)
    e = float(contact_model.restitution_coeff)
    mu_s = float(contact_model.mu_s)
    mu_d = float(contact_model.mu_d)
    mu_r = float(contact_model.rolling_friction_coeff)
    dt = float(getattr(contact_model, "dt", 0.0) or 0.0)

    # Эффективная масса пары m_eff = mi*mj/(mi+mj) (как в ContactModel /
    # CPU-Numba ядре). Защита от деления на ноль.
    m_eff = (mass[:, None] * mass[None, :]) / cp.where(
        (mass[:, None] + mass[None, :]) > 0.0,
        (mass[:, None] + mass[None, :]), 1.0)

    # Демпфирование с учётом эффективной массы (Cundall & Strack), идентично
    # ContactModel._damping_coefficient. Если e<=0 — критическое демпфирование.
    if e <= 0.0:
        gamma_n = -2.0 * cp.sqrt(kn * m_eff)
        gamma_t = -2.0 * cp.sqrt(kt * m_eff)
    else:
        ln_e = math.log(e)
        denom = math.sqrt(math.pi ** 2 + ln_e * ln_e)
        gamma_n = -2.0 * ln_e * cp.sqrt(kn * m_eff) / denom
        gamma_t = -2.0 * ln_e * cp.sqrt(kt * m_eff) / denom

    fn_scalar = kn * overlap + gamma_n * overlap_rate
    fnx = fn_scalar * nx
    fny = fn_scalar * ny

    # Касательное смещение накапливается как v_t * dt (как в CPU-ядре).
    # Используем кумулятивный буфер, переданный вызывающей стороной (numpy),
    # конвертируем в cupy, обновляем и пишем обратно.
    if tangential_disp is not None:
        td = cp.asarray(np.asarray(tangential_disp, dtype=cp.float64))
        td_update = td + tvx * dt
    else:
        td = cp.zeros((n, n), dtype=cp.float64)
        td_update = tvx * dt

    # ft_trial = -kt*td - gamma_t*v_t, Кулоновский предел mu*|Fn|
    abs_fn = cp.abs(fn_scalar)
    ft_trial = -kt * td_update - gamma_t * tvx
    mu_abs = mu_s * abs_fn
    ft_scalar = cp.where(
        ft_trial > mu_abs,
        -mu_d * abs_fn,
        cp.where(ft_trial < -mu_abs, mu_d * abs_fn, ft_trial),
    )

    # Касательный вектор для силы: f_t = ft * (-ny, nx) -> на i действует -ft_vec,
    # на j — +ft_vec (как в CPU-ядре).
    ftx = -ft_scalar * ny
    fty = ft_scalar * nx

    # Силы на i от контакта (i<j): -fn - ft ; на j: +fn + ft.
    # Матрица выше диагонали (i<j) даёт на i: -(fnx+ftx), на j: +(fnx+ftx).
    force_x_mat = -(fnx + ftx)  # для i<j действует на i
    force_y_mat = -(fny + fty)
    # Симметрично раскрываем: полная матрица = M - M.T
    force_x_full = force_x_mat - force_x_mat.T
    force_y_full = force_y_mat - force_y_mat.T

    # Момент качения: r_eff = ri*rj/(ri+rj), omega_rel = ang_i - ang_j
    r_eff = (
        radius[:, None] * radius[None, :]
        / cp.where((radius[:, None] + radius[None, :]) > 0.0,
                   radius[:, None] + radius[None, :], 1.0)
    )
    omega_rel = ang_vel[:, None] - ang_vel[None, :]
    sign_om = cp.sign(omega_rel)
    roll_torque = -mu_r * abs_fn * r_eff * sign_om  # на i, для i<j
    torque_full = roll_torque - roll_torque.T

    # Обнуляем неактивные контакты
    force_x_full = cp.where(mask, force_x_full, 0.0)
    force_y_full = cp.where(mask, force_y_full, 0.0)
    torque_full = cp.where(mask, torque_full, 0.0)

    # Записываем обновлённое касательное смещение обратно (если буфер передан)
    if tangential_disp is not None:
        np.asarray(tangential_disp)[:] = cp.asnumpy(td_update)

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
