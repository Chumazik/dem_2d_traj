"""Батчевое Numba-ядро для контактов частица‑частица."""

import numpy as np
from numba import njit, prange


@njit(cache=True, fastmath=True, parallel=True)
def _pairwise_particle_forces(
    pos,             # (N, 2)
    vel,             # (N, 2)
    ang_vel,         # (N,)
    radius,          # (N,)
    force_out,       # (N, 2) – куда суммировать
    torque_out,      # (N,)   – куда суммировать
    tangential_disp, # (N, N) – накопительное касательное смещение
    kn,
    gamma_n,
    kt,
    gamma_t,
    mu_s,
    mu_d,
    rolling_friction,
    dt,
):
    """Батчевая версия part-part контактов.

    Эквивалентна двойному циклу из
    :func:`dem.force_calculation.compute_all_forces`, но без Python-overhead
    и аллокаций промежуточных ``np.ndarray``.
    """
    N = pos.shape[0]
    for i in prange(N):
        xi = pos[i, 0]
        yi = pos[i, 1]
        vxi = vel[i, 0]
        vyi = vel[i, 1]
        ri = radius[i]
        fi0 = 0.0
        fi1 = 0.0
        ti = 0.0
        for j in range(i + 1, N):
            dx = pos[j, 0] - xi
            dy = pos[j, 1] - yi
            rsum = ri + radius[j]
            dist2 = dx * dx + dy * dy
            if dist2 >= rsum * rsum:
                continue
            dist = np.sqrt(dist2)
            if dist < 1e-12:
                continue
            inv_dist = 1.0 / dist
            overlap = rsum - dist
            nx = dx * inv_dist
            ny = dy * inv_dist

            # относительная скорость
            rvx = vel[j, 0] - vxi
            rvy = vel[j, 1] - vyi
            overlap_rate = rvx * nx + rvy * ny

            # касательное смещение (накопительно) вдоль касательного направления.
            # Раньше здесь было 0.0*dt (трение не накапливалось) - исправлено:
            # тангенциальное смещение растёт как v_t * dt.
            # Касательный вектор (2D): t = (-ny, nx), тангенц. скорость = rv·t.
            tvx = rvx * (-ny) + rvy * nx
            td = tangential_disp[i, j] + tvx * dt
            tangential_disp[i, j] = td

            # нормальная сила
            fn_scalar = kn * overlap + gamma_n * overlap_rate
            fnx = fn_scalar * nx
            fny = fn_scalar * ny

            # касательная сила (Кулоновский предел)
            # Согласовано с ContactModel.compute_forces:
            # ft_trial = -kt*td - gamma_t*v_t (упругое + вязкое демпфирование).
            ft_trial = -kt * td - gamma_t * tvx
            abs_fn = fn_scalar if fn_scalar >= 0.0 else -fn_scalar
            mu_abs = mu_s * abs_fn
            if ft_trial > mu_abs:
                ft_scalar = -mu_d * abs_fn
            elif ft_trial < -mu_abs:
                ft_scalar = mu_d * abs_fn
            else:
                ft_scalar = ft_trial

            # тангенциальный вектор: перпендикуляр к нормали (2D)
            ftx = -ft_scalar * ny
            fty = ft_scalar * nx

            fi0 += -(fnx + ftx)
            fi1 += -(fny + fty)
            force_out[j, 0] += fnx + ftx
            force_out[j, 1] += fny + fty

            # момент качения
            rj = radius[j]
            r_eff = (ri * rj) / (ri + rj)
            omega_rel = ang_vel[i] - ang_vel[j]
            if omega_rel > 0.0:
                sign_omega = 1.0
            elif omega_rel < 0.0:
                sign_omega = -1.0
            else:
                sign_omega = 0.0
            rolling_torque = -rolling_friction * abs_fn * r_eff * sign_omega
            ti += rolling_torque
            torque_out[j] += -rolling_torque

        force_out[i, 0] += fi0
        force_out[i, 1] += fi1
        torque_out[i] += ti
