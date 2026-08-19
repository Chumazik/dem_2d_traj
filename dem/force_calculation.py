"""Расчёт сил DEM и шаг интегратора.

В модуле три публичных пути:

* :func:`compute_all_forces` – основной путь, делегирующий парные контакты
  батчевому :func:`compute_pairwise_forces` (Numba или CuPy) и оставляющий
  контакты частица‑граница на CPU из‑за разнородной геометрии
  (``WallLine`` / ``WallCircle`` / ``Lifter``);
* :func:`compute_pairwise_forces` – высокопроизводительный батчевый путь
  для парных контактов частица‑частица. Автовыбор бэкенда:
    1. ``config.use_gpu=True`` и доступный CuPy → :func:`dem.gpu_backend.compute_pairwise_forces_cupy`;
    2. иначе → Numba-версия из :mod:`dem.jit_kernels.pairwise`;
    3. иначе → чистый Python (вызывающая сторона может перейти на
       :func:`_pairwise_python_loop`);
* :func:`velocity_verlet_step` – шаг интегратора. Автовыбор бэкенда:
  CuPy → Numba → чистый Python (аналогично).
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Dict, List

from .contact_model import ContactModel, Contact
from .geometry import WallCircle  # Добавлено импорт WallCircle
from .jit_kernels.pairwise import _pairwise_particle_forces as _nb_pairwise
from .jit_kernels.integrator import _velocity_verlet_step as _nb_verlet
from .spatial_grid import create_spatial_grid, SpatialGrid


# ---------------------------------------------------------------------------
# Хелперы для выбора бэкенда
# ---------------------------------------------------------------------------
def _want_gpu(contact_model: ContactModel) -> bool:
    """``True``, если в конфигурации запрошен GPU-бэкенд."""
    cfg = getattr(contact_model, "config", None)
    if cfg is None:
        return False
    return bool(getattr(cfg, "use_gpu", False))


def _want_jit(contact_model: ContactModel) -> bool:
    cfg = getattr(contact_model, "config", None)
    if cfg is None:
        return True
    return bool(getattr(cfg, "use_jit", True))


# ---------------------------------------------------------------------------
# CPU / чистый Python (обратная совместимость)
# ---------------------------------------------------------------------------
def _pairwise_python_loop(particles, contact_model: ContactModel,
                           contacts=None) -> None:
    """O(N) чистый Python с пространственным хешированием — для пар без GPU/Numba."""
    if contacts is None:
        contacts = {}
    
    # Создаем пространственную сетку для ускорения поиска контактов
    grid = create_spatial_grid(particles, particles[0].radius if particles else 0.05)
    
    for i, pi in enumerate(particles):
        # Получаем только частицы из соседних ячеек
        neighbors = grid.get_neighboring_particles(pi)
        
        for pj in neighbors:
            # Избегаем проверки частицы с самой собой и дубликатов (i < j)
            if pj.id <= pi.id:
                continue
                
            delta = pj.pos - pi.pos
            dist = float(np.linalg.norm(delta))

            if dist >= pi.radius + pj.radius:
                continue

            overlap = pi.radius + pj.radius - dist
            if dist == 0.0:
                normal = np.array([1.0, 0.0])
                inv_dist = 0.0
            else:
                normal = delta / dist
                inv_dist = 1.0 / dist

            rel_vel = pj.vel - pi.vel
            overlap_rate = float(np.dot(rel_vel, normal))
            # Касательный вектор (2D): перпендикуляр к нормали.
            tangent = np.array([-normal[1], normal[0]], dtype=float)
            rel_vel_tang = float(np.dot(rel_vel, tangent))  # тангенц. скорость (скаляр)

            contact_key = (pi.id, pj.id)
            if contact_key not in contacts:
                contacts[contact_key] = Contact(pi.id, pj.id)
            contact = contacts[contact_key]
            # Накопление касательного смещения вдоль касательного направления.
            tangential_displacement = (
                contact.tangential_displacement
                + rel_vel_tang * contact_model.dt
            )
            effective_radius = (pi.radius * pj.radius) / (pi.radius + pj.radius)

            fn_vec, ft_vec, torque_i, torque_j = contact_model.compute_forces(
                overlap, overlap_rate,
                tangential_displacement,
                rel_vel_tang,
                effective_radius, normal, pi, pj,
            )

            pi.apply_force(-fn_vec - ft_vec, torque_i)
            pj.apply_force(fn_vec + ft_vec, torque_j)
            contact.tangential_displacement = tangential_displacement

    return contacts


# ---------------------------------------------------------------------------
# Главная точка входа: CPU/Numba/GPU авто-выбор для пар, CPU для границ
# ---------------------------------------------------------------------------
def compute_all_forces(particles, boundaries, contact_model: ContactModel,
                       contacts=None):
    """Полный расчёт сил: парные → автовыбор бэкенда; частица‑граница → CPU;
    гравитация → +Y.

    Контакты частица‑граница остаются на CPU из‑за разнородной геометрии.
    Возвращает словарь контактов (пополняется :class:`dem.contact_model.Contact`
    только для граничных пар).

    Силовые аккумуляторы частиц (``force`` / ``torque``) обнуляются в начале —
    это единственная точка, гарантирующая, что каждый вызов возвращает
    чистые силы текущего шага (контакты + гравитация). Без сброса остатки
    прошлого шага (например, от интегратора Verlet) накапливались бы с
    гравитацией.
    """
    if contacts is None:
        contacts = {}

    for p in particles:
        p.reset_force()

    # Буфер касательного смещения, общий для Numba- и GPU-путей (накапливается
    # между шагами, как в ContactModel.compute_forces). Хранится на contact_model,
    # чтобы не терять состояние между вызовами step().
    if not hasattr(contact_model, "tangential_disp") or contact_model.tangential_disp is None:
        contact_model.tangential_disp = np.zeros((len(particles), len(particles)), dtype=np.float64)
    elif contact_model.tangential_disp.shape != (len(particles), len(particles)):
        contact_model.tangential_disp = np.zeros((len(particles), len(particles)), dtype=np.float64)
    tangential_disp = contact_model.tangential_disp

    # --- Парные контакты: автовыбор GPU → Numba → Python ---
    if len(particles) >= 2:
        if _want_gpu(contact_model):
            try:
                from . import gpu_backend
                if gpu_backend.is_available():
                    gpu_backend.compute_pairwise_forces_cupy(particles, contact_model, tangential_disp)
                    for i in range(len(particles)):
                        for j in range(i + 1, len(particles)):
                            key = (particles[i].id, particles[j].id)
                            if key not in contacts:
                                contacts[key] = Contact(particles[i].id, particles[j].id)
            except Exception:
                if _want_jit(contact_model):
                    try:
                        compute_pairwise_forces(particles, contact_model, tangential_disp)
                        for i in range(len(particles)):
                            for j in range(i + 1, len(particles)):
                                key = (particles[i].id, particles[j].id)
                                if key not in contacts:
                                    contacts[key] = Contact(particles[i].id, particles[j].id)
                    except Exception:
                        _pairwise_python_loop(particles, contact_model, contacts)
                else:
                    _pairwise_python_loop(particles, contact_model, contacts)
        elif _want_jit(contact_model):
            try:
                compute_pairwise_forces(particles, contact_model)
                for i in range(len(particles)):
                    for j in range(i + 1, len(particles)):
                        key = (particles[i].id, particles[j].id)
                        if key not in contacts:
                            contacts[key] = Contact(particles[i].id, particles[j].id)
            except Exception:
                _pairwise_python_loop(particles, contact_model, contacts)
        else:
            _pairwise_python_loop(particles, contact_model, contacts)

    # --- Контакты частица-граница (всегда CPU) ---
    for boundary in boundaries:
        for p in particles:
            coll = boundary.detect_collision(p)
            if coll is None:
                continue

            overlap, contact_point, normal, overlap_rate, tangential_velocity = coll
            # Calculate tangential velocity component (dot product with tangent vector)
            tangent_vector = np.array([-normal[1], normal[0]])
            rel_vel_tang = float(np.dot(tangential_velocity, tangent_vector))

            contact_key = (p.id, boundary)
            if contact_key not in contacts:
                contacts[contact_key] = Contact(p.id, None)
            contact = contacts[contact_key]
            tangential_displacement = (
                contact.tangential_displacement
                + rel_vel_tang * contact_model.dt
            )

            fn_vec, ft_vec, torque_p, _ = contact_model.compute_forces(
                overlap, overlap_rate,
                tangential_displacement,
                rel_vel_tang, p.radius, normal, p, None,
            )

            # Calculate torque from tangential force (r × Ft)
            torque_arm = contact_point - p.pos  # Vector from particle center to contact point
            torque_from_friction = torque_arm[0] * ft_vec[1] - torque_arm[1] * ft_vec[0]
            
            # Apply forces: the normal force should push the particle away from the boundary
            # For WallCircle: normal points outward from drum center, so -fn_vec pushes inward
            # For WallLine: normal points into valid region, so fn_vec pushes away from wall
            if isinstance(boundary, WallCircle):
                # WallCircle normal points outward, so we need to reverse the normal force
                p.apply_force(-fn_vec + ft_vec, torque_p + torque_from_friction)
            else:
                # WallLine normal points into valid region, so normal force is correct
                p.apply_force(fn_vec + ft_vec, torque_p + torque_from_friction)
            if isinstance(boundary, WallCircle):
                boundary.apply_driving_torque(torque_p)

    # --- Гравитация (внешняя сила, действует вдоль +Y) ---
    cfg = getattr(contact_model, "config", None)
    g_value = getattr(cfg, "gravity", 9.81) if cfg is not None else 9.81
    if g_value is not None and g_value != 0.0:
        for p in particles:
            p.force[1] += p.mass * float(g_value)

    return contacts


# ---------------------------------------------------------------------------
# Высокопроизводительный путь: только парные контакты (Numba/CuPy)
# ---------------------------------------------------------------------------
def compute_pairwise_forces(
    particles,
    contact_model: ContactModel,
    tangential_disp: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Считает силы/моменты парных контактов батчево.

    Поведение зависит от ``contact_model.config``:

    * ``use_gpu=True`` + доступный CuPy → :func:`dem.gpu_backend.compute_pairwise_forces_cupy`;
    * иначе + ``use_jit=True`` (по умолчанию) → Numba-ядро
      :func:`dem.jit_kernels.pairwise._pairwise_particle_forces`;
    * иначе → чистый Python O(N) с пространственным хешированием.

    Поля ``force``/``torque`` частиц должны быть обнулены до вызова.
    Возвращает обновлённый массив ``tangential_disp``.
    """
    if not particles:
        return tangential_disp

    # Для небольшого числа частиц (N < 100) используем O(N²) напрямую
    # для лучшей производительности (меньше накладных расходов на сетку)
    if len(particles) < 100:
        # ---- GPU (CuPy) ----
        if _want_gpu(contact_model):
            try:
                from . import gpu_backend
                if gpu_backend.is_available():
                    gpu_backend.compute_pairwise_forces_cupy(particles, contact_model, tangential_disp)
                    return tangential_disp
            except Exception:
                # мягкий фолбэк
                pass

        # ---- Numba / batch ----
        if _want_jit(contact_model):
            try:
                return _compute_pairwise_forces_numba(particles, contact_model, tangential_disp)
            except Exception:
                # мягкий фолбэк
                pass

        # ---- Чистый Python ----
        return _compute_pairwise_forces_python(particles, contact_model, tangential_disp)
    
    # Для большого числа частиц (N >= 100) используем пространственное хеширование
    # в чистом Python пути, а для GPU/Numba используем полные массивы
    # так как они оптимизированы для батчевой обработки
    
    # ---- GPU и Numba пути используют полные массивы ----
    if _want_gpu(contact_model) or _want_jit(contact_model):
        # Для GPU и Numba используем существующий код без изменений
        if _want_gpu(contact_model):
            try:
                from . import gpu_backend
                if gpu_backend.is_available():
                    gpu_backend.compute_pairwise_forces_cupy(particles, contact_model, tangential_disp)
                    return tangential_disp
            except Exception:
                pass
        
        if _want_jit(contact_model):
            try:
                return _compute_pairwise_forces_numba(particles, contact_model, tangential_disp)
            except Exception:
                pass
    
    # ---- Чистый Python путь использует пространственное хеширование ----
    return _compute_pairwise_forces_python(particles, contact_model, tangential_disp)
    
    # Параметры контактной модели
    kn = float(contact_model.kn)
    kt = float(contact_model.kt)
    restitution = float(contact_model.restitution_coeff)
    mu_s = float(contact_model.mu_s)
    mu_d = float(contact_model.mu_d)
    rf = float(contact_model.rolling_friction_coeff)
    dt = float(getattr(contact_model, "dt", 0.0) or 0.0)

    # ---- GPU (CuPy) ----
    if _want_gpu(contact_model):
        try:
            from . import gpu_backend
            if gpu_backend.is_available():
                # Для GPU используем полный массив частиц, так как GPU ядро
                # ожидает полный набор данных для батчевой обработки
                gpu_backend.compute_pairwise_forces_cupy(particles, contact_model, tangential_disp)
                return tangential_disp
        except Exception:
            # мягкий фолбэк
            pass

    # ---- Numba / batch ----
    if _want_jit(contact_model):
        try:
            # Для Numba используем полный набор данных, так как ядро оптимизировано
            # для полной матрицы контактов и использует prange для параллелизма
            return _compute_pairwise_forces_numba(particles, contact_model, tangential_disp)
        except Exception:
            # мягкий фолбэк
            pass

    # ---- Чистый Python ----
    # Для чистого Python уже используется пространственное хеширование в _pairwise_python_loop
    return _compute_pairwise_forces_python(particles, contact_model, tangential_disp)


def _compute_pairwise_forces_numba(
    particles, contact_model: ContactModel,
    tangential_disp: Optional[np.ndarray],
) -> np.ndarray:
    n = len(particles)
    pos = np.empty((n, 2), dtype=np.float64)
    vel = np.empty((n, 2), dtype=np.float64)
    ang_vel = np.empty(n, dtype=np.float64)
    radius = np.empty(n, dtype=np.float64)
    mass = np.empty(n, dtype=np.float64)
    for i, p in enumerate(particles):
        pos[i] = p.pos
        vel[i] = p.vel
        ang_vel[i] = p.ang_vel
        radius[i] = p.radius
        mass[i] = p.mass

    force_out = np.zeros((n, 2), dtype=np.float64)
    torque_out = np.zeros(n, dtype=np.float64)

    if tangential_disp is None or tangential_disp.shape != (n, n):
        tangential_disp = np.zeros((n, n), dtype=np.float64)

    kn = float(contact_model.kn)
    kt = float(contact_model.kt)
    restitution = float(contact_model.restitution_coeff)
    mu_s = float(contact_model.mu_s)
    mu_d = float(contact_model.mu_d)
    rf = float(contact_model.rolling_friction_coeff)
    dt = float(getattr(contact_model, "dt", 0.0) or 0.0)

    _nb_pairwise(
        pos, vel, ang_vel, radius, mass,
        force_out, torque_out,
        tangential_disp,
        kn, kt, restitution,
        mu_s, mu_d, rf, dt,
    )

    for i, p in enumerate(particles):
        p.force += force_out[i]
        p.torque += float(torque_out[i])

    return tangential_disp


def _compute_pairwise_forces_python(
    particles, contact_model: ContactModel,
    tangential_disp: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    """Чистый Python O(N²) — для пар без GPU/Numba; понимает tangential_disp
    как вход/выход Numba-стиля (накапливает в квадратной матрице)."""
    n = len(particles)
    if tangential_disp is None or tangential_disp.shape != (n, n):
        tangential_disp = np.zeros((n, n), dtype=np.float64)

    dt = float(getattr(contact_model, "dt", 0.0) or 0.0)
    for i in range(n):
        pi = particles[i]
        for j in range(i + 1, n):
            pj = particles[j]
            delta = pj.pos - pi.pos
            dist = float(np.linalg.norm(delta))
            radius_sum = float(pi.radius + pj.radius)
            if dist >= radius_sum:
                continue
            overlap = radius_sum - dist
            if dist == 0.0:
                normal = np.array([1.0, 0.0])
                inv_dist = 0.0
            else:
                normal = delta / dist
                inv_dist = 1.0 / dist
            rel = pj.vel - pi.vel
            overlap_rate = float(np.dot(rel, normal))
            # Касательный вектор (2D).
            tangent = np.array([-normal[1], normal[0]], dtype=float)
            rel_vel_tang = float(np.dot(rel, tangent))

            # Накопление касательного смещения вдоль касательного направления.
            tangential_disp[i, j] += rel_vel_tang * dt
            tangential_disp[j, i] = -tangential_disp[i, j]

            r_eff = (float(pi.radius) * float(pj.radius)) / radius_sum

            fn_vec, ft_vec, torque_i, torque_j = contact_model.compute_forces(
                overlap, overlap_rate,
                tangential_disp[i, j],
                rel_vel_tang,
                r_eff, normal, pi, pj,
            )

            pi.apply_force(-fn_vec - ft_vec, torque_i)
            pj.apply_force(fn_vec + ft_vec, torque_j)

    return tangential_disp


# ---------------------------------------------------------------------------
# Шаг интегратора: автовыбор GPU/Numba/Python
# ---------------------------------------------------------------------------
def velocity_verlet_step(
    particles, contact_model: ContactModel, dt: float
) -> None:
    """Один шаг Velocity Verlet (полушаг скоростей + шаг позиций).

    Автовыбор бэкенда:
    1. ``use_gpu=True`` + CuPy → :func:`dem.gpu_backend.velocity_verlet_step_cupy`;
    2. ``use_jit=True`` (по умолчанию) → Numba-ядро
       :func:`dem.jit_kernels.integrator._velocity_verlet_step`;
    3. иначе → чистый Python.

    Поля ``force``/``torque`` уже должны быть посчитаны текущим шагом;
    после шага они НЕ сбрасываются (делает вызывающая сторона).
    """
    if not particles:
        return

    # ---- GPU ----
    if _want_gpu(contact_model):
        try:
            from . import gpu_backend
            if gpu_backend.is_available():
                gpu_backend.velocity_verlet_step_cupy(particles, dt)
                return
        except Exception:
            pass

    # ---- Numba ----
    if _want_jit(contact_model):
        try:
            _velocity_verlet_step_numba(particles, dt)
            return
        except Exception:
            pass

    # ---- Чистый Python ----
    _velocity_verlet_step_python(particles, dt)


def _velocity_verlet_step_numba(particles, dt: float) -> None:
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


def _velocity_verlet_step_python(particles, dt: float) -> None:
    for p in particles:
        a = p.force / p.mass
        alpha = p.torque / p.inertia
        p.vel += 0.5 * a * dt
        p.ang_vel += 0.5 * alpha * dt

    for p in particles:
        p.pos += p.vel * dt
