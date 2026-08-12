import unittest

import numpy as np

from dem.contact_model import ContactModel
from dem.force_calculation import compute_all_forces
from dem.particle import Particle
from utils.config import SimulationConfig


class TestGravity(unittest.TestCase):
    """Гравитация должна применяться к свободным частицам как
    постоянная внешняя сила вдоль +Y (направление «вниз» в канвасе)."""

    def _free_particle(self, x=0.0, y=0.0, mass=1.0):
        return Particle(
            id=0, radius=0.02, density=2500.0,
            mass=mass, inertia=0.5 * mass * (0.02 ** 2),
            pos=np.array([x, y]), vel=np.array([0.0, 0.0]),
            ang_vel=0.0, force=np.zeros(2), torque=0.0,
            history=[],
        )

    def test_default_gravity_applied(self):
        p = self._free_particle(mass=2.0)
        cfg = SimulationConfig()
        cm = ContactModel(kn=1e5, restitution_coeff=0.9, config=cfg)
        # Сбросить контакты, частица свободна.
        compute_all_forces([p], [], cm)
        # После compute_all_forces: сила от гравитации m·g = 2.0·9.81 = 19.62 вдоль +Y.
        self.assertAlmostEqual(p.force[0], 0.0, places=8)
        self.assertAlmostEqual(p.force[1], 2.0 * 9.81, places=8)

    def test_zero_gravity_no_force(self):
        p = self._free_particle(mass=2.0)
        cfg = SimulationConfig(gravity=0.0)
        cm = ContactModel(kn=1e5, restitution_coeff=0.9, config=cfg)
        compute_all_forces([p], [], cm)
        self.assertAlmostEqual(p.force[0], 0.0, places=8)
        self.assertAlmostEqual(p.force[1], 0.0, places=8)

    def test_config_without_gravity_field_falls_back_to_default(self):
        """Объект конфигурации, у которого нет поля gravity (например, чужой код) — фолбэк."""
        p = self._free_particle(mass=1.5)

        class _CfgNoGravity:
            pass

        cm = ContactModel(kn=1e5, restitution_coeff=0.9, config=_CfgNoGravity())
        compute_all_forces([p], [], cm)
        self.assertAlmostEqual(p.force[1], 1.5 * 9.81, places=8)

    def test_contacts_dont_change_with_free_particles(self):
        p1 = self._free_particle(x=0.0, mass=1.0)
        p2 = self._free_particle(x=1.5, mass=1.0)  # далеко, нет контакта
        cfg = SimulationConfig()
        cm = ContactModel(kn=1e5, restitution_coeff=0.9, config=cfg)
        contacts = compute_all_forces([p1, p2], [], cm)
        # Пар записей контактов (для обратной совместимости) — 1 пара.
        self.assertEqual(len(contacts), 1)
        self.assertAlmostEqual(p1.force[1], 9.81, places=8)
        self.assertAlmostEqual(p2.force[1], 9.81, places=8)


if __name__ == "__main__":
    unittest.main()
