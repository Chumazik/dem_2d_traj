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
    dt: float = 1e-5                       # с
    total_time: float = 5.0                # с