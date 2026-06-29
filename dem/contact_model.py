import numpy as np
from typing import Optional, Tuple

class Contact:
    """Класс для хранения состояния контакта между двумя частицами."""
    
    def __init__(self, id1: int, id2: int):
        self.id1 = id1
        self.id2 = id2
        self.tangential_displacement = 0.0

class ContactModel:
    """Модель контакта с упругой, демпфирующей и трением."""

    def __init__(self,
                 kn: float,
                 restitution_coeff: float = 0.9,
                 mu_s: float = 0.5,
                 mu_d: float = 0.4,
                 rolling_friction_coeff: float = 0.01,
                 kt: Optional[float] = None,
                 dt: Optional[float] = None,
                 config: Optional[object] = None):   # <-- добавлен параметр config
        """
        Параметры:
            kn – жёсткость нормального контакта.
            restitution_coeff – коэффициент восстановления.
            mu_s – статическое трение.
            mu_d – динамическое трение.
            rolling_friction_coeff – коэффициент качения.
            kt – жёсткость касательного контакта (по умолчанию 2/7 * kn).
            dt – шаг интегрирования, нужен для расчёта касательного смещения.
            config – объект конфигурации (SimulationConfig), может использоваться внешними модулями.
        """
        self.kn = kn
        self.kt = kt if kt is not None else 2.0 * kn / 7.0
        self.restitution_coeff = restitution_coeff
        self.mu_s = mu_s
        self.mu_d = mu_d
        self.rolling_friction_coeff = rolling_friction_coeff
        self.dt = dt if dt is not None else 0.0
        self.config = config   # <-- сохраняем конфигурацию

    def compute_forces(self,
                       overlap: float,
                       overlap_rate: float,
                       tangential_displacement: float,
                       rel_vel_tang: float,
                       effective_radius: float,
                       normal_unit_vector: np.ndarray,
                       particle1,
                       particle2: Optional[object] = None) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        Возвращает:
            normal_force_vector,
            tangential_force_vector,
            rolling_torque_on_particle1,
            rolling_torque_on_particle2 (0, если граница)
        """
        # ---------- Нормальная сила ----------
        gamma_n = -2 * np.sqrt(self.kn * self.restitution_coeff)
        gamma_t = gamma_n

        fn_scalar = self.kn * overlap + gamma_n * overlap_rate
        normal_force_vector = fn_scalar * normal_unit_vector

        # ---------- Касательная сила ----------
        ft_trial = -self.kt * tangential_displacement - gamma_t * rel_vel_tang

        if np.abs(ft_trial) > self.mu_s * np.abs(fn_scalar):
            sign = np.sign(rel_vel_tang) if rel_vel_tang != 0 else 0.0
            ft_scalar = -self.mu_d * np.abs(fn_scalar) * sign
        else:
            ft_scalar = ft_trial

        tangential_force_vector = ft_scalar * np.array([-normal_unit_vector[1], normal_unit_vector[0]])

        # ---------- Момент качения ----------
        if particle2 is not None:
            r_eff = (particle1.radius * particle2.radius) / (particle1.radius + particle2.radius)
            omega_rel = particle1.ang_vel - particle2.ang_vel
        else:
            r_eff = particle1.radius
            omega_rel = particle1.ang_vel

        rolling_torque1 = -self.rolling_friction_coeff * np.abs(fn_scalar) * r_eff * np.sign(omega_rel)
        rolling_torque2 = 0.0 if particle2 is None else -rolling_torque1

        return normal_force_vector, tangential_force_vector, rolling_torque1, rolling_torque2

    def update_contact(self, contact: Contact, rel_vel_tang: float, dt: float):
        """Обновляет касательное смещение для контакта."""
        contact.tangential_displacement += rel_vel_tang * dt
