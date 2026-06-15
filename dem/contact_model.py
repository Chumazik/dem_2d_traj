import numpy as np

class ContactModel:

    def __init__(self, kn, kt=None, restitution_coeff=0.9, mu_s=0.5, mu_d=0.4, rolling_friction_coeff=0.01):

        self.kn = kn

        self.kt = kt if kt is not None else 2 * kn / 7

        # Вычисление коэффициентов демпфирования

        self.gamma_n = -2 * np.log(restitution_coeff) * np.sqrt(kn * 1) / np.pi  # Для упрощения assume m_eff = 1

        self.gamma_t = -2 * np.log(restitution_coeff) * np.sqrt(self.kt * 1) / np.pi  # Для упрощения assume m_eff = 1

        self.mu_s = mu_s


        self.mu_d = mu_d


        self.rolling_friction_coeff = rolling_friction_coeff


    def compute_forces(self, overlap, overlap_rate, tangential_displacement, rel_vel_tang, effective_radius, normal_unit_vector, particle1, particle2=None):



        """Compute contact forces and torques."""



        # Нормальная сила

        normal_force = self.kn * overlap + self.gamma_n * overlap_rate



        normal_force_vector = normal_force * normal_unit_vector







        # Касательная сила

        ft_trial = -self.kt * tangential_displacement - self.gamma_t * rel_vel_tang



        if np.abs(ft_trial) > self.mu_s * np.abs(normal_force):

            tangential_force = -self.mu_d * np.abs(normal_force) * (rel_vel_tang / np.abs(rel_vel_tang))

        else:

            tangential_force = ft_trial


        # Момент сопротивления качению

        rolling_torque_on_particle1 = 0



        if particle2 is not None:

            r_eff = (particle1.radius * particle2.radius) / (particle1.radius + particle2.radius)

            omega_rel = particle1.ang_vel - particle2.ang_vel

            rolling_torque_on_particle1 = -self.rolling_friction_coeff * np.abs(normal_force) * r_eff * np.sign(omega_rel)

        else:

            # Для частицы и границы

            r_eff = particle1.radius

            omega_rel = particle1.ang_vel

            rolling_torque_on_particle1 = -self.rolling_friction_coeff * np.abs(normal_force) * r_eff * np.sign(omega_rel)

        return normal_force_vector, tangential_force, rolling_torque_on_particle1, 0 if particle2 is None else -rolling_torque_on_particle1