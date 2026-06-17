import numpy as np
from dataclasses import dataclass, field
from typing import List

from .particle import Particle
from .contact_model import ContactModel
from .geometry import WallCircle
from .integrator import velocity_verlet_step
from utils.config import SimulationConfig

@dataclass
class Simulation:
    """Главный класс симуляции."""
    config: SimulationConfig
    particles: List[Particle] = field(default_factory=list)
    boundaries: List[object] = field(default_factory=list)
    contact_model: ContactModel = None
    time: List[float] = field(default_factory=list)
    torque_history: List[float] = field(default_factory=list)

    def __post_init__(self):
        self.contact_model = ContactModel(
            kn=self.config.kn,
            restitution_coeff=self.config.restitution_coeff,
            mu_s=self.config.friction_static,
            mu_d=self.config.friction_dynamic,
            rolling_friction_coeff=self.config.rolling_friction
        )
        self.initialize_particles()
        self.initialize_boundaries()

    # --------------------------------------------------------------------- #
    # Инициализация
    # --------------------------------------------------------------------- #
    def initialize_particles(self):
        """Размещает частицы внутри барабана без перекрытий (простая спираль)."""
        R = self.config.drum_radius - self.config.particle_radius - 1e-4
        angle_step = 2 * np.pi / max(self.config.num_particles, 1)
        radius_step = self.config.particle_radius * 2.2

        count = 0
        r = self.config.particle_radius
        while count < self.config.num_particles and r < R:
            n_on_ring = int(2 * np.pi * r / (2 * self.config.particle_radius * 1.1))
            for k in range(n_on_ring):
                if count >= self.config.num_particles:
                    break
                theta = k * 2 * np.pi / n_on_ring
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                pos = np.array([x, y])
                mass = self.config.particle_density * np.pi * (self.config.particle_radius ** 2)
                inertia = 0.5 * mass * (self.config.particle_radius ** 2)
                particle = Particle(
                    id=count,
                    radius=self.config.particle_radius,
                    density=self.config.particle_density,
                    mass=mass,
                    inertia=inertia,
                    pos=pos,
                    vel=np.zeros(2),
                    ang_vel=0.0,
                    history=[pos.copy()]
                )
                self.particles.append(particle)
                count += 1
            r += radius_step

        # Если после спирали не удалось разместить все частицы, заполняем случайными позициями
        while count < self.config.num_particles:
            angle = np.random.rand() * 2 * np.pi
            rad = np.random.rand() * (R - self.config.particle_radius) + self.config.particle_radius
            pos = np.array([rad * np.cos(angle), rad * np.sin(angle)])
            # простая проверка на перекрытие
            if any(np.linalg.norm(pos - p.pos) < 2 * self.config.particle_radius for p in self.particles):
                continue
            mass = self.config.particle_density * np.pi * (self.config.particle_radius ** 2)
            inertia = 0.5 * mass * (self.config.particle_radius ** 2)
            particle = Particle(
                id=count,
                radius=self.config.particle_radius,
                density=self.config.particle_density,
                mass=mass,
                inertia=inertia,
                pos=pos,
                vel=np.zeros(2),
                ang_vel=0.0,
                history=[pos.copy()]
            )
            self.particles.append(particle)
            count += 1

    def initialize_boundaries(self):
        """Создаёт единственную границу – вращающийся барабан."""
        self.boundaries.append(
            WallCircle(
                center=(0.0, 0.0),
                radius=self.config.drum_radius,
                omega=self.config.drum_omega
            )
        )

    # --------------------------------------------------------------------- #
    # Шаг и запуск
    # --------------------------------------------------------------------- #
    def step(self):
        """Выполняет один временной шаг и сохраняет реактивный момент."""
        velocity_verlet_step(self.particles, self.config.dt, self.contact_model, self.boundaries)

        # Сохраняем реактивный момент от барабана (если он есть)
        drum = next((b for b in self.boundaries if isinstance(b, WallCircle)), None)
        if drum is not None:
            self.torque_history.append(-drum.applied_torque)  # знак противоположный реактивному
            drum.applied_torque = 0.0  # сбрасываем для следующего шага

    def run(self):
        """Запускает симуляцию до достижения total_time."""
        t = 0.0
        self.time.clear()
        self.torque_history.clear()
        while t < self.config.total_time:
            self.step()
            self.time.append(t)
            t += self.config.dt

    def get_trajectories(self):
        """Возвращает список историй всех частиц."""
        return [p.history for p in self.particles]