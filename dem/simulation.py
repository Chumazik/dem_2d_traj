import numpy as np
from dataclasses import dataclass, field
from typing import List

from .particle import Particle
from .contact_model import ContactModel, compute_all_forces  # Исправлен импорт compute_all_forces
from .geometry import WallCircle
from .integrator import velocity_verlet_step
from config import SimulationConfig  # Исправлен импорт SimulationConfig

@dataclass
class Simulation:
    """Главный класс симуляции."""
    config: SimulationConfig
    particles: List[Particle] = field(default_factory=list)
    boundaries: List[object] = field(default_factory=list)
    contact_model: ContactModel = None
    time: List[float] = field(default_factory=list)
    torque_history: List[float] = field(default_factory=list)
    contacts: dict = field(default_factory=dict)  # Добавлен атрибут для хранения контактов
    stop_requested: bool = False  # Флаг для прерывания симуляции

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
        """Размещает частицы внутри барабана без перекрытий (метод hexagonal packing)."""
        R = self.config.drum_radius - self.config.particle_radius - 1e-4
        particle_diameter = 2 * self.config.particle_radius

        # Вычисляем количество рядов и столбцов частиц
        num_rows = int(np.sqrt(R**2 / (3 * particle_diameter**2)))
        num_cols = int(num_rows)

        count = 0
        for i in range(num_rows):
            for j in range(num_cols):
                if count >= self.config.num_particles:
                    break
                x_offset = 0.5 * particle_diameter * (i % 2)
                x = x_offset + j * particle_diameter
                y = np.sqrt(3) / 2 * i * particle_diameter
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

        # Если после hexagonal packing не удалось разместить все частицы, заполняем случайными позициями
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
        self.contacts = compute_all_forces(self.particles, self.boundaries, self.contact_model, self.contacts)

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
        step_count = 0
        while t < self.config.total_time and not self.stop_requested:
            self.step()
            self.time.append(t)
            t += self.config.dt
            step_count += 1
            if step_count % 10 == 0:
                # Здесь можно добавить сигнал или колбэк для обновления прогресса
                print(f"Progress: {t / self.config.total_time * 100:.2f}%")  # Пример вывода в консоль

    def stop(self):
        """Запрашивает остановку симуляции."""
        self.stop_requested = True

    def get_trajectories(self):
        """Возвращает список историй всех частиц."""
        return [p.history for p in self.particles]
