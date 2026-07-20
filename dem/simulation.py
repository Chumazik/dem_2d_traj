import numpy as np
from dataclasses import dataclass, field
from typing import List

from .particle import Particle
from .contact_model import ContactModel, Contact
from .geometry import WallCircle, Lifter
from .integrator import velocity_verlet_step
from .force_calculation import compute_all_forces
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
    contacts: dict = field(default_factory=dict)
    stop_requested: bool = False

    def __post_init__(self):
        self.contact_model = ContactModel(
            kn=self.config.kn,
            restitution_coeff=self.config.restitution_coeff,
            mu_s=self.config.friction_static,
            mu_d=self.config.friction_dynamic,
            rolling_friction_coeff=self.config.rolling_friction,
            dt=self.config.dt,
            config=self.config
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

        while count < self.config.num_particles:
            angle = np.random.rand() * 2 * np.pi
            rad = np.random.rand() * (R - self.config.particle_radius) + self.config.particle_radius
            pos = np.array([rad * np.cos(angle), rad * np.sin(angle)])
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
        """Создаёт границу – вращающийся барабан и лифтеры."""
        drum = WallCircle(
            center=(0.0, 0.0),
            radius=self.config.drum_radius,
            omega=self.config.drum_omega
        )
        self.boundaries.append(drum)
        
        if self.config.num_lifters > 0 and self.config.lifter_height > 0:
            for i in range(self.config.num_lifters):
                base_angle = 2 * np.pi * i / self.config.num_lifters
                lifter = Lifter(
                    drum_center=(0.0, 0.0),
                    drum_radius=self.config.drum_radius,
                    height=self.config.lifter_height,
                    width=self.config.lifter_width,
                    base_angle=base_angle,
                    omega=self.config.drum_omega
                )
                self.boundaries.append(lifter)

    # --------------------------------------------------------------------- #
    # Шаг и запуск
    # --------------------------------------------------------------------- #
    def step(self):
        """Выполняет один временной шаг и сохраняет реактивный момент.

        Последовательность:
            1. Обновить текущий угол лифтеров/барабана (если есть).
            2. Пересчитать силы и касательные смещения для всех контактов.
            3. Сделать шаг Velocity Verlet (полушаг скоростей + позиции).
            4. Обнулить накопленные ``force``/``torque`` (для следующего шага).
            5. Снять показания реактивного момента с границ.
        """
        current_time = self.time[-1] if self.time else 0.0
        for b in self.boundaries:
            if hasattr(b, 'update_time'):
                b.update_time(current_time)

        self.contacts = compute_all_forces(
            self.particles, self.boundaries, self.contact_model, self.contacts
        )

        velocity_verlet_step(
            self.particles, self.config.dt, self.contact_model, self.boundaries
        )

        total_torque = 0.0
        for b in self.boundaries:
            if hasattr(b, 'applied_torque'):
                total_torque += b.applied_torque
                b.applied_torque = 0.0
                
        self.torque_history.append(-total_torque)

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
                print(f"Progress: {t / self.config.total_time * 100:.2f}%")

    def stop(self):
        """Запрашивает остановку симуляции."""
        self.stop_requested = True

    def get_trajectories(self):
        """Возвращает список историй всех частиц."""
        return [p.history for p in self.particles]
