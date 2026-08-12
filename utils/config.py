"""Конфигурационные параметры симуляции DEM.

Все параметры задаются в системе СИ. Датакласс `SimulationConfig` хранит
значения по умолчанию, которые могут быть переопределены через GUI.
"""

from dataclasses import dataclass

@dataclass
class SimulationConfig:
    """Параметры симуляции.

    При необходимости пользователь может изменить любые поля через GUI.
    """
    num_particles: int = 100
    particle_radius: float = 0.02          # м
    particle_density: float = 2500.0       # кг/м³
    kn: float = 1e5                        # N/m
    restitution_coeff: float = 0.9         # коэффициент восстановления
    friction_static: float = 0.5           # μ_s
    friction_dynamic: float = 0.4          # μ_d
    rolling_friction: float = 0.01         # μ_r
    drum_radius: float = 0.5                # м
    drum_omega: float = 2.0                 # рад/с
    lifter_height: float = 0.0              # м (высота лифтера, 0 - нет лифтеров)
    lifter_width: float = 0.02              # м (ширина лифтера)
    num_lifters: int = 0                    # количество лифтеров
    dt: float = 1e-5                       # с
    total_time: float = 5.0                # с
    use_jit: bool = True                   # Numba-JIT интегратор (иначе чистый Python)
    use_gpu: bool = False                  # GPU (CuPy) для парных контактов и Verlet (фолбэк: Numba → CPU)
    gravity: float = 9.81                  # м/с², действует вдоль +Y (направление «вниз» в системе координат канваса)
    # Параметры начальной упаковки частиц:
    gap_fraction: float = 0.05             # доля дополнительного зазора между центрами частиц:  spacing = d·(1+gap)
    angle_of_repose_deg: float = 35.0      # угол естественного откоса для верхней границы «кучи»
    apparent_mill_filling: float = 28.0    # заполнение, % (определяет ширину слоя на дне)
