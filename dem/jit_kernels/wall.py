"""Батчевое Numba-ядро для контактов частица‑стенка (заготовка)."""

import numpy as np
from numba import njit, prange


@njit(cache=True, fastmath=True, parallel=True)
def _wall_particle_forces(
    pos,             # (N, 2)
    vel,             # (N, 2)
    ang_vel,         # (N,)
    radius,          # (N,)
    force_out,       # (N, 2) – куда суммировать
    torque_out,      # (N,)   – куда суммировать
    kn,
    gamma_n,
    kt,
    gamma_t,
    mu_s,
    mu_d,
    rolling_friction,
):
    """Батчевая версия part-wall контактов для одного ``WallLine``.

    Заготовка для будущей Numba-реализации. На данный момент
    вычислительный путь стенок реализован в чистом Python в
    :mod:`dem.force_calculation`, поскольку геометрия границ
    разнородна (``WallLine`` / ``WallCircle`` / ``Lifter``).

    Параметры стенки задаются скалярами ``wall_point_x``, ``wall_point_y``,
    ``wall_normal_x``, ``wall_normal_y`` (нормаль уже нормирована).
    """
    N = pos.shape[0]
    wpx = 0.0
    wpy = 0.0
    wnx = 0.0
    wny = 0.0
    for i in prange(N):
        xi = pos[i, 0]
        yi = pos[i, 1]
        vxi = vel[i, 0]
        vyi = vel[i, 1]
        ri = radius[i]

        dist = (xi - wpx) * wnx + (yi - wpy) * wny
        if dist > 0.0:
            continue  # частица снаружи

        overlap = ri - dist
        # нормаль, направленная от стенки к частице
        nx = -wnx
        ny = -wny
        # скорость сближения по нормали
        overlap_rate = vxi * nx + vyi * ny

        fn_scalar = kn * overlap + gamma_n * overlap_rate
        abs_fn = fn_scalar if fn_scalar >= 0.0 else -fn_scalar

        # в этой 2D-модели касательное смещение для стенки тоже ~0
        ft_trial = -kt * 0.0
        mu_abs = mu_s * abs_fn
        if ft_trial > mu_abs:
            ft_scalar = -mu_d * abs_fn
        elif ft_trial < -mu_abs:
            ft_scalar = mu_d * abs_fn
        else:
            ft_scalar = ft_trial

        ftx = -ft_scalar * ny
        fty = ft_scalar * nx
        fnx = fn_scalar * nx
        fny = fn_scalar * ny

        force_out[i, 0] += -(fnx + ftx)
        force_out[i, 1] += -(fny + fty)

        # момент качения от стенки: r_eff = ri
        omega_rel = ang_vel[i]
        if omega_rel > 0.0:
            sign_omega = 1.0
        elif omega_rel < 0.0:
            sign_omega = -1.0
        else:
            sign_omega = 0.0
        rolling_torque = -rolling_friction * abs_fn * ri * sign_omega
        torque_out[i] += rolling_torque
