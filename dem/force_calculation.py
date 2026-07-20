"""Расчёт сил DEM с поддержкой нескольких бэкендов.

В модуле три публичных пути:

* :func:`compute_all_forces` – основной CPU-путь (чистый Python, без
  Numba). Сохраняет полную совместимость с предыдущими версиями;
* :func:`compute_pairwise_forces` – высокопроизводительный путь для
  парных контактов частица‑частица, который автоматически выбирает
  бэкенд:
  - если в конфигурации ``contact_model.config.use_gpu`` истинно и
    :mod:`dem.gpu_backend` доступен → CuPy-реализация;
  - иначе → батчевая Numba-версия из :mod:`dem.jit_kernels.pairwise`;
* :func:`velocity_verlet_step` – шаг интегратора с тем же выбором
  бэкенда, что и в :func:`compute_pairwise_forces` (CuPy или Numba).

Контакты частица‑граница всегда считаются на CPU из-за разнородной
геометрии (``WallLine`` / ``WallCircle`` / ``Lifter``).
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from .contact_model import ContactModel, Contact
from .geometry import WallCircle  # Добавлено импорт WallCircle
from .jit_kernels.pairwise import _pairwise_particle_forces as _nb_pairwise
from .jit_kernels.integrator import _velocity_verlet_step as _nb_verlet


# ---------------------------------------------------------------------------
# CPU / чистый Python (обратная совместимость)
# ---------------------------------------------------------------------------

def compute_all_forces(particles, boundaries, contact_model: ContactModel, contacts=None):
    """
    Вычисляет все взаимодействия частиц‑частиц и частица‑граница.
    Сохраняет реактивный момент в объекте WallCircle.
    """
    n = len(particles)

    if contacts is None:
        contacts = {}

    # ---- Частица-частица ----
    for i in range(n):
        for j in range(i + 1, n):
            pi, pj = particles[i], particles[j]
            delta = pj.pos - pi.pos
            dist = np.linalg.norm(delta)

            if dist < pi.radius + pj.radius:
                overlap = pi.radius + pj.radius - dist
                normal = delta / dist if dist != 0 else np.array([1.0, 0.0])

                # относительная скорость в нормальном и касательном направлениях
                rel_vel = pj.vel - pi.vel
                overlap_rate = np.dot(rel_vel, normal)
                tangential_velocity = rel_vel - overlap_rate * normal

                # Создаем или обновляем контакт
                contact_key = (pi.id, pj.id)
                if contact_key not in contacts:
                    contacts[contact_key] = Contact(pi.id, pj.id)

                contact = contacts[contact_key]
                # Используем dt, хранящийся в модели контакта
                tangential_displacement = (
                    contact.tangential_displacement
                    + np.dot(tangential_velocity, normal) * contact_model.dt
                )
                effective_radius = (pi.radius * pj.radius) / (pi.radius + pj.radius)

                fn_vec, ft_vec, torque_i, torque_j = contact_model.compute_forces(
                    overlap,
                    overlap_rate,
                    tangential_displacement,
                    np.dot(tangential_velocity, normal),
                    effective_radius,
                    normal,
                    pi,
                    pj
                )

                pi.apply_force(-fn_vec - ft_vec, torque_i)
                pj.apply_force(fn_vec + ft_vec, torque_j)

                # Обновляем касательное смещение
                contact.tangential_displacement = tangential_displacement

    # ---- Частица-граница ----
    for boundary in boundaries:
        for p in particles:
            coll = boundary.detect_collision(p)
            if coll is None:
                continue

            overlap, contact_point, normal, overlap_rate, tangential_velocity = coll

            # относительная касательная скорость (модуль)
            rel_vel_tang = np.dot(tangential_velocity, normal)

            # Создаем или обновляем контакт
            contact_key = (p.id, boundary)
            if contact_key not in contacts:
                contacts[contact_key] = Contact(p.id, None)

            contact = contacts[contact_key]
            tangential_displacement = (
                contact.tangential_displacement
                + rel_vel_tang * contact_model.dt
            )

            fn_vec, ft_vec, torque_p, _ = contact_model.compute_forces(
                overlap,
                overlap_rate,
                tangential_displacement,
                rel_vel_tang,
                p.radius,
                normal,
                p,
                None
            )

            p.apply_force(-fn_vec - ft_vec, torque_p)

            if isinstance(boundary, WallCircle):
                # реактивный момент, который частицы передали барабану
                boundary.apply_driving_torque(torque_p)

    return contacts


# ---------------------------------------------------------------------------
# Высокопроизводительный бэкенд part-part контактов
# ---------------------------------------------------------------------------

def _want_gpu(contact_model: ContactModel) -> bool:
    """Возвращает ``True``, если в конфигурации запрошен GPU-бэкенд."""
    cfg = getattr(contact_model, "config", None)
    if cfg is None:
        return False
    return bool(getattr(cfg, "use_gpu", False))


def compute_pairwise_forces(
    particles,
    contact_model: ContactModel,
    tangential_disp: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Считает силы/моменты парных контактов.

    Автоматически выбирает бэкенд:

    * ``contact_model.config.use_gpu`` и доступный CuPy → :mod:`dem.gpu_backend`;
    * иначе → батчевая Numba-версия из :mod:`dem.jit_kernels.pairwise`.

    Перед вызовом поля ``force``/``torque`` частиц должны быть обнулены.
    Возвращает обновлённый массив ``tangential_disp`` (Numba-путь)
    или ``None`` (GPU-путь).
    """
    if not particles:
        return tangential_disp

    use_gpu = _want_gpu(contact_model)
    if use_gpu:
        try:
            from . import gpu_backend
            if gpu_backend.is_available():
                gpu_backend.compute_pairwise_forces_cupy(particles, contact_model)
                return tangential_disp
        except Exception:
            # мягкий фолбэк на Numba при любой ошибке GPU-пути
            pass

    # ---- Numba / batch ----
    n = len(particles)
    pos = np.empty((n, 2), dtype=np.float64)
    vel = np.empty((n, 2), dtype=np.float64)
    ang_vel = np.empty(n, dtype=np.float64)
    radius = np.empty(n, dtype=np.float64)
    for i, p in enumerate(particles):
        pos[i] = p.pos
        vel[i] = p.vel
        ang_vel[i] = p.ang_vel
        radius[i] = p.radius

    force_out = np.zeros((n, 2), dtype=np.float64)
    torque_out = np.zeros(n, dtype=np.float64)

    if tangential_disp is None or tangential_disp.shape != (n, n):
        tangential_disp = np.zeros((n, n), dtype=np.float64)

    kn = float(contact_model.kn)
    gamma_n = -2.0 * np.sqrt(kn * float(contact_model.restitution_coeff))
    kt = float(contact_model.kt)
    gamma_t = gamma_n
    mu_s = float(contact_model.mu_s)
    mu_d = float(contact_model.mu_d)
    rf = float(contact_model.rolling_friction_coeff)
    dt = float(getattr(contact_model, "dt", 0.0) or 0.0)

    _nb_pairwise(
        pos, vel, ang_vel, radius,
        force_out, torque_out,
        tangential_disp,
        kn, gamma_n, kt, gamma_t,
        mu_s, mu_d, rf, dt,
    )

    for i, p in enumerate(particles):
        p.force += force_out[i]
        p.torque += float(torque_out[i])

    return tangential_disp


# ---------------------------------------------------------------------------
# Шаг интегратора с автовыбором бэкенда
# ---------------------------------------------------------------------------

def velocity_verlet_step(particles, contact_model: ContactModel, dt: float) -> None:
    """Один шаг Velocity Verlet (полушаг скоростей + шаг позиций).

    Бэкенд выбирается так же, как и в :func:`compute_pairwise_forces`:
    если ``contact_model.config.use_gpu`` истинно и CuPy доступен →
    считаем на GPU, иначе → батчевая Numba-версия из
    :mod:`dem.jit_kernels.integrator`.

    Перед вызовом поля ``force``/``torque`` частиц уже должны быть
    посчитаны текущим шагом. После шага ``force``/``torque`` НЕ
    обнуляются – это делает вызывающая сторона перед следующим
    пересчётом сил.
    """
    if not particles:
        return

    use_gpu = _want_gpu(contact_model)
    if use_gpu:
        try:
            from . import gpu_backend
            if gpu_backend.is_available():
                gpu_backend.velocity_verlet_step_cupy(particles, dt)
                return
        except Exception:
            pass

    # ---- Numba / batch ----
    n = len(particles)
    pos = np.empty((n, 2), dtype=np.float64)
    vel = np.empty((n, 2), dtype=np.float64)
    ang_vel = np.empty(n, dtype=np.float64)
    force = np.empty((n, 2), dtype=np.float64)
    torque = np.empty(n, dtype=np.float64)
    mass = np.empty(n, dtype=np.float64)
    inertia = np.empty(n, dtype=np.float64)

    for i, p in enumerate(particles):
        pos[i] = p.pos
        vel[i] = p.vel
        ang_vel[i] = p.ang_vel
        force[i] = p.force
        torque[i] = p.torque
        mass[i] = p.mass
        inertia[i] = p.inertia

    _nb_verlet(pos, vel, ang_vel, force, torque, mass, inertia, dt)

    for i, p in enumerate(particles):
        p.pos = pos[i]
        p.vel = vel[i]
        p.ang_vel = float(ang_vel[i])
