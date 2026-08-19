"""GPU-бэкенд для горячего цикла DEM (CuPy) с постоянным состоянием на GPU.

Основные особенности:
- Постоянное хранение состояния частиц на GPU (позиции, скорости, силы)
- Объединённое ядро для расчёта сил и интеграции (избегает лишних синхронизаций)
- Zero-copy режим для минимизации передачи данных CPU↔GPU
- Синхронизация с CPU только для визуализации (раз в N шагов)

Если CuPy недоступен или CUDA-устройств нет, :func:`is_available`
возвращает ``False``. Вызывающая сторона (``dem.force_calculation``)
обрабатывает это и прозрачно переключается на путь Numba или CPU.
"""

from __future__ import annotations

import math
from typing import Dict, List, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .contact_model import ContactModel
    from .particle import Particle

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


class PersistentGPUState:
    """Постоянное состояние частиц на GPU для минимизации передачи CPU↔GPU."""

    def __init__(self, particles: List['Particle'], contact_model: 'ContactModel'):
        """
        Инициализирует постоянное состояние на GPU.
        
        Args:
            particles: Список частиц для симуляции
            contact_model: Модель контактов для параметров
        """
        if not _cp or not is_available():
            raise RuntimeError("CuPy GPU backend unavailable")
        
        self.cp = _cp
        self.n = len(particles)
        self.contact_model = contact_model
        
        # Выделяем память на GPU для состояния частиц
        self.pos = self.cp.empty((self.n, 2), dtype=self.cp.float64)
        self.vel = self.cp.empty((self.n, 2), dtype=self.cp.float64)
        self.ang_vel = self.cp.empty(self.n, dtype=self.cp.float64)
        self.force = self.cp.zeros((self.n, 2), dtype=self.cp.float64)
        self.torque = self.cp.zeros(self.n, dtype=self.cp.float64)
        self.radius = self.cp.empty(self.n, dtype=self.cp.float64)
        self.mass = self.cp.empty(self.n, dtype=self.cp.float64)
        self.inertia = self.cp.empty(self.n, dtype=self.cp.float64)
        
        # Инициализируем состояние из частиц
        self._upload_particles(particles)
        
        # Буфер для касательного смещения (общий для всех шагов)
        self.tangential_disp = self.cp.zeros((self.n, self.n), dtype=self.cp.float64)
        
        # Компилируем ядро
        self._compile_kernel()

    def _upload_particles(self, particles: List['Particle']) -> None:
        """Загружает состояние частиц на GPU."""
        for i, p in enumerate(particles):
            self.pos[i, 0] = float(p.pos[0])
            self.pos[i, 1] = float(p.pos[1])
            self.vel[i, 0] = float(p.vel[0])
            self.vel[i, 1] = float(p.vel[1])
            self.ang_vel[i] = float(p.ang_vel)
            self.force[i, 0] = float(p.force[0])
            self.force[i, 1] = float(p.force[1])
            self.torque[i] = float(p.torque)
            self.radius[i] = float(p.radius)
            self.mass[i] = float(p.mass)
            self.inertia[i] = float(p.inertia)

    def _download_particles(self, particles: List['Particle']) -> None:
        """Скачивает состояние частиц с GPU на CPU."""
        # Скачиваем только необходимые данные для визуализации
        pos_np = self.cp.asnumpy(self.pos)
        vel_np = self.cp.asnumpy(self.vel)
        ang_vel_np = self.cp.asnumpy(self.ang_vel)
        
        for i, p in enumerate(particles):
            p.pos[0] = float(pos_np[i, 0])
            p.pos[1] = float(pos_np[i, 1])
            p.vel[0] = float(vel_np[i, 0])
            p.vel[1] = float(vel_np[i, 1])
            p.ang_vel = float(ang_vel_np[i])

    def _compile_kernel(self) -> None:
        """Компилирует объединённое GPU-ядро для сил и интеграции."""
        # Параметры контактной модели
        kn = float(self.contact_model.kn)
        kt = float(self.contact_model.kt)
        e = float(self.contact_model.restitution_coeff)
        mu_s = float(self.contact_model.mu_s)
        mu_d = float(self.contact_model.mu_d)
        rf = float(self.contact_model.rolling_friction_coeff)
        dt = float(getattr(self.contact_model, "dt", 0.0) or 0.0)
        
        # Компилируем ядро с этими параметрами
        self._kernel = self._create_combined_kernel(
            kn, kt, e, mu_s, mu_d, rf, dt
        )

    def _create_combined_kernel(self, kn, kt, e, mu_s, mu_d, rf, dt):
        """Создаёт и компилирует объединённое GPU-ядро."""
        kernel_code = f"""
        extern "C" __global__
        void combined_dem_kernel(
            const double* pos, const double* vel, const double* ang_vel,
            const double* radius, const double* mass, const double* inertia,
            double* force, double* torque, double* tangential_disp,
            int n, double dt
        ) {{
            int i = blockDim.x * blockIdx.x + threadIdx.x;
            if (i >= n) return;
            
            // --- Полушаг скоростей (первая половина Velocity Verlet) ---
            double inv_m = 1.0 / mass[i];
            double inv_I = 1.0 / inertia[i];
            double ax = force[i] * inv_m;
            double ay = force[i + n] * inv_m;
            double alpha = torque[i] * inv_I;
            
            vel[i] += 0.5 * ax * dt;
            vel[i + n] += 0.5 * ay * dt;
            ang_vel[i] += 0.5 * alpha * dt;
            
            // --- Обновление позиций ---
            pos[i] += vel[i] * dt;
            pos[i + n] += vel[i + n] * dt;
            
            // --- Сброс сил ---
            force[i] = 0.0;
            force[i + n] = 0.0;
            torque[i] = 0.0;
            
            // --- Расчёт парных контактов (O(N²) на GPU) ---
            for (int j = i + 1; j < n; j++) {{
                double dx = pos[j] - pos[i];
                double dy = pos[j + n] - pos[i + n];
                double rsum = radius[i] + radius[j];
                double dist2 = dx*dx + dy*dy;
                
                if (dist2 >= rsum*rsum) continue;
                
                double dist = sqrt(dist2);
                double inv_dist = (dist > 0.0) ? 1.0 / dist : 0.0;
                double overlap = rsum - dist;
                
                // Нормальный вектор
                double nx = dx * inv_dist;
                double ny = dy * inv_dist;
                
                // Относительная скорость
                double rvx = vel[j] - vel[i];
                double rvy = vel[j + n] - vel[i + n];
                double overlap_rate = rvx * nx + rvy * ny;
                
                // Касательный вектор и скорость
                double tvx = rvx * (-ny) + rvy * nx;
                
                // Эффективная масса
                double m_eff = (mass[i] * mass[j]) / (mass[i] + mass[j]);
                
                // Демпфирование
                double gamma_n, gamma_t;
                if ({e} <= 0.0) {{
                    gamma_n = -2.0 * sqrt({kn} * m_eff);
                    gamma_t = -2.0 * sqrt({kt} * m_eff);
                }} else {{
                    double ln_e = log({e});
                    double denom = sqrt(3.141592653589793 * 3.141592653589793 + ln_e * ln_e);
                    gamma_n = -2.0 * ln_e * sqrt({kn} * m_eff) / denom;
                    gamma_t = -2.0 * ln_e * sqrt({kt} * m_eff) / denom;
                }}
                
                // Нормальная сила
                double fn_scalar = {kn} * overlap + gamma_n * overlap_rate;
                double fnx = fn_scalar * nx;
                double fny = fn_scalar * ny;
                
                // Касательное смещение
                double td = tangential_disp[i * n + j] + tvx * {dt};
                tangential_disp[i * n + j] = td;
                tangential_disp[j * n + i] = -td;
                
                // Касательная сила
                double ft_trial = -{kt} * td - gamma_t * tvx;
                double abs_fn = fabs(fn_scalar);
                double mu_abs = {mu_s} * abs_fn;
                double ft_scalar = (ft_trial > mu_abs) ? -{mu_d} * abs_fn : 
                                  (ft_trial < -mu_abs) ? {mu_d} * abs_fn : ft_trial;
                
                double ftx = -ft_scalar * ny;
                double fty = ft_scalar * nx;
                
                // Применяем силы (Newton's 3rd law)
                atomicAdd(&force[i], -fnx - ftx);
                atomicAdd(&force[i + n], -fny - fty);
                atomicAdd(&force[j], fnx + ftx);
                atomicAdd(&force[j + n], fny + fty);
                
                // Момент качения
                double r_eff = (radius[i] * radius[j]) / rsum;
                double omega_rel = ang_vel[i] - ang_vel[j];
                double sign_om = (omega_rel > 0.0) ? 1.0 : (omega_rel < 0.0) ? -1.0 : 0.0;
                double rolling_torque = -{rf} * abs_fn * r_eff * sign_om;
                
                atomicAdd(&torque[i], rolling_torque);
                atomicAdd(&torque[j], -rolling_torque);
            }}
            
            // --- Второй полушаг скоростей (Velocity Verlet) ---
            inv_m = 1.0 / mass[i];
            inv_I = 1.0 / inertia[i];
            ax = force[i] * inv_m;
            ay = force[i + n] * inv_m;
            alpha = torque[i] * inv_I;
            
            vel[i] += 0.5 * ax * dt;
            vel[i + n] += 0.5 * ay * dt;
            ang_vel[i] += 0.5 * alpha * dt;
        }}
        """
        
        # Компилируем ядро
        return self.cp.RawKernel(kernel_code, 'combined_dem_kernel')

    def step(self, particles: List['Particle'], sync_every: int = 10) -> None:
        """
        Выполняет один шаг симуляции на GPU.
        
        Args:
            particles: Список частиц (для синхронизации с CPU)
            sync_every: Синхронизировать с CPU каждые N шагов (для визуализации)
        """
        # Запускаем ядро
        threads_per_block = 256
        blocks = (self.n + threads_per_block - 1) // threads_per_block
        
        self._kernel(
            (blocks,), (threads_per_block,),
            (self.pos, self.vel, self.ang_vel, self.radius, self.mass, self.inertia,
             self.force, self.torque, self.tangential_disp, self.n, dt)
        )
        
        # Синхронизируем с CPU только когда нужно (для визуализации)
        if sync_every > 0 and self.step_count % sync_every == 0:
            self._download_particles(particles)
        
        self.step_count += 1

    def sync(self, particles: List['Particle']) -> None:
        """Принудительная синхронизация состояния GPU→CPU."""
        self._download_particles(particles)


# ---------------------------------------------------------------------------
# Парные контактные силы
# ---------------------------------------------------------------------------
# Глобальный кэш для постоянного состояния GPU
_gpu_state_cache: Dict[int, PersistentGPUState] = {}

def _get_gpu_state(particles: List['Particle'], contact_model: 'ContactModel') -> PersistentGPUState:
    """Получает или создаёт постоянное состояние GPU для данных частиц."""
    # Используем хэш от количества частиц и параметров модели как ключ
    key = (len(particles), contact_model.kn, contact_model.kt, contact_model.restitution_coeff)
    if key not in _gpu_state_cache:
        _gpu_state_cache[key] = PersistentGPUState(particles, contact_model)
    return _gpu_state_cache[key]

def compute_pairwise_forces_cupy(particles, contact_model, tangential_disp=None) -> None:
    """Батчевая парная нормальная+касательная сила + момент качения на GPU.

    Использует постоянное состояние на GPU для минимизации передачи CPU↔GPU.
    """
    if not _cp or not is_available():
        raise RuntimeError("CuPy GPU backend unavailable")
    
    if not particles:
        return
    
    # Получаем постоянное состояние GPU
    gpu_state = _get_gpu_state(particles, contact_model)
    
    # Обновляем силы на GPU (без копирования данных)
    gpu_state.step(particles, sync_every=0)  # Не синхронизируем автоматически
    
    # Обновляем касательное смещение в переданном буфере (если есть)
    if tangential_disp is not None:
        np.asarray(tangential_disp)[:] = _cp.asnumpy(gpu_state.tangential_disp)

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

