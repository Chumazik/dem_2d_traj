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
        """Размещает частицы внутри барабана в виде осевшей кучи на дне.

        Использует тот же алгоритм, что и статический превью в web/static/script.js:

        * центры частиц лежат в шахматной гексагональной упаковке по рядам;
        * межчастичный зазор: ``spacing = d·(1+gap_fraction)`` (по умолчанию
          +5 % к диаметру). Это даёт видимое «не приклеены друг к другу»;
        * нижняя граница кучи: ``y = +effective_R`` (дно барабана в +Y);
        * ширина ряда убывает линейно с высотой по углу естественного откоса
          (``angle_of_repose_deg``);
        * ширина основания кучи зависит от ``apparent_mill_filling``;
        * начальная скорость: ``vel = 0``, ``ang_vel = 0`` (тяжёлая куча
          оседает преимущественно без движения).

        Если запрошено больше частиц, чем помещается в кучу, добавляем
        их равномерно в доступных слотах, а при переполнении случайным
        образом (с проверкой непересечения).
        """
        import math

        R_full = self.config.drum_radius
        r = self.config.particle_radius
        d = 2 * r
        spacing = d * (1.0 + self.config.gap_fraction)
        row_h = spacing * math.sqrt(3.0) / 2.0

        effective_R = R_full - r  # радиус для центров частиц

        repose_rad = math.radians(self.config.angle_of_repose_deg)
        sin_r = math.sin(repose_rad)
        cos_r = math.cos(repose_rad)
        safe_sin_over_cos = sin_r / max(cos_r, 1e-6)

        fill_frac = max(0.05, min(0.6, self.config.apparent_mill_filling / 100.0))
        bed_radius = effective_R * (0.4 + 0.5 * math.sqrt(fill_frac / 0.5))

        count = 0
        row = 0
        placed_coords: list[np.ndarray] = []
        while count < self.config.num_particles:
            y_base = effective_R - row_h * row - r
            # Если верх достиг верхней кромки эффективной области — выход из
            # основного цикла (дополнительные частицы будут размещены
            # случайно/сверху).
            if y_base < -effective_R + r:
                break

            y_above_bottom = (effective_R - y_base) / max(1e-6, effective_R)
            half_width = bed_radius * max(0.0, 1.0 - y_above_bottom * safe_sin_over_cos)
            row_offset = spacing / 2.0 if (row % 2) else 0.0

            # Проходим по ряду с шагом spacing, центы выровнены в шахмат.
            x_centers = np.arange(
                -half_width - row_offset,
                half_width + 1e-9,
                spacing,
            )
            for x in x_centers:
                if count >= self.config.num_particles:
                    break
                if x * x + y_base * y_base > effective_R * effective_R:
                    continue
                pos = np.array([x, y_base], dtype=float)
                self._add_particle_at(pos, id=count)
                placed_coords.append(pos.copy())
                count += 1
            row += 1

        # Если в осевшей куче поместилось меньше, чем запрошено, дополняем
        # случайной упаковкой в оставшемся пространстве барабана (для редких
        # случаев, когда num_particles значительно больше capacity кучи).
        attempts = 0
        while count < self.config.num_particles and attempts < 5000:
            attempts += 1
            angle = np.random.rand() * 2 * np.pi
            rad = np.random.rand() * (effective_R - r) + r
            pos = np.array([rad * math.cos(angle), rad * math.sin(angle)],
                           dtype=float)
            if any(np.linalg.norm(pos - p) < d * (1.0 + 0.5 * self.config.gap_fraction)
                   for p in placed_coords):
                continue
            self._add_particle_at(pos, id=count)
            placed_coords.append(pos.copy())
            count += 1

    def _add_particle_at(self, pos: np.ndarray, id: int) -> None:
        """Создаёт частицу в ``pos`` с нулевой скоростью/угловой скоростью и
        стартовым history из одной точки."""
        mass = self.config.particle_density * np.pi * (self.config.particle_radius ** 2)
        inertia = 0.5 * mass * (self.config.particle_radius ** 2)
        particle = Particle(
            id=id,
            radius=self.config.particle_radius,
            density=self.config.particle_density,
            mass=mass,
            inertia=inertia,
            pos=pos,
            vel=np.zeros(2),
            ang_vel=0.0,
            history=[pos.copy()],
        )
        self.particles.append(particle)

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
            1. Продвинуть внутреннее время шага на ``config.dt`` и обновить
               угол лифтеров/барабана (если есть).
            2. Пересчитать силы и касательные смещения для всех контактов.
            3. Сделать шаг Velocity Verlet (полушаг скоростей + позиции).
            4. Обнулить накопленные ``force``/``torque`` (для следующего шага).
            5. Снять показания реактивного момента с границ.
        """
        # Продвигаем внутреннее время так, чтобы лифтеры корректно вращались
        # и при запуске через внешний цикл (run_simulation), и при прямом
        # вызове step() в тестах. ``self.time`` остаётся историей для
        # потребителей (графики, /partial_results).
        if not hasattr(self, "_sim_time") or self._sim_time is None:
            self._sim_time = 0.0
        self._sim_time += self.config.dt
        current_time = self._sim_time

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
        self.time.append(current_time)

    def run(self):
        """Запускает симуляцию до достижения total_time."""
        # ``Simulation.step`` теперь сам управляет ``self._sim_time``
        # и добавляет точку в ``self.time``. Здесь только выполняем цикл.
        self.time.clear()
        if hasattr(self, "_sim_time"):
            self._sim_time = 0.0
        # Сбрасываем текущие углы лифтеров на base, чтобы первый шаг
        # отсчитывал время от нуля заново.
        for b in self.boundaries:
            if hasattr(b, "base_angle"):
                b.current_angle = b.base_angle
                b._update_corners()
        self.torque_history.clear()
        step_count = 0
        while step_count * self.config.dt < self.config.total_time and not self.stop_requested:
            self.step()
            step_count += 1
            if step_count % 10 == 0:
                print(f"Progress: {(step_count * self.config.dt) / self.config.total_time * 100:.2f}%")

    def stop(self):
        """Запрашивает остановку симуляции."""
        self.stop_requested = True

    def get_trajectories(self):
        """Возвращает список историй всех частиц."""
        return [p.history for p in self.particles]
