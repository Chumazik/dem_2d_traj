import sys
import types
import unittest
from unittest.mock import patch


class TestGpuBackendAvailability(unittest.TestCase):

    def test_is_available_returns_false_without_cupy(self):
        """Если cupy не импортирован, is_available() должно возвращать False."""
        import importlib
        from dem import gpu_backend
        importlib.reload(gpu_backend)
        # В тестовой среде cupy, как правило, отсутствует.
        if gpu_backend._cp is None:
            self.assertFalse(gpu_backend.is_available())
        else:
            # cupy установлен → зависит от CUDA
            self.assertIsInstance(gpu_backend.is_available(), bool)

    def test_import_error_surfaced(self):
        """Если cupy был недоступен при импорте, import_error() хранит исключение."""
        from dem import gpu_backend
        err = gpu_backend.import_error()
        if gpu_backend._cp is None:
            self.assertIsNotNone(err)
        else:
            self.assertIsNone(err)


class TestForceCalculationBackendSelection(unittest.TestCase):

    def test_default_flag_is_cpu_jit(self):
        from dem.force_calculation import _want_gpu, _want_jit
        from utils.config import SimulationConfig
        from dem.contact_model import ContactModel

        cfg = SimulationConfig()  # по умолчанию use_jit=True, use_gpu=False
        cm = ContactModel(kn=1.0, restitution_coeff=0.9, mu_s=0.5,
                         mu_d=0.4, rolling_friction_coeff=0.0,
                         dt=1e-5, config=cfg)
        self.assertTrue(_want_jit(cm))
        self.assertFalse(_want_gpu(cm))

    def test_use_gpu_flag(self):
        from dem.force_calculation import _want_gpu
        from utils.config import SimulationConfig
        from dem.contact_model import ContactModel

        cfg = SimulationConfig(use_gpu=True)
        cm = ContactModel(kn=1.0, restitution_coeff=0.9, config=cfg)
        self.assertTrue(_want_gpu(cm))

    def test_no_config_falls_back_to_defaults(self):
        from dem.force_calculation import _want_gpu, _want_jit
        from dem.contact_model import ContactModel

        cm = ContactModel(kn=1.0, restitution_coeff=0.9)  # без config
        self.assertTrue(_want_jit(cm))
        self.assertFalse(_want_gpu(cm))


class TestForceCalculationWithGpuMocked(unittest.TestCase):
    """End-to-end test compute_all_forces с мок-апом GPU-бэкенда."""

    def _setup(self, call_log):
        # Подсовываем фейковый gpu_backend, который помечает свой вызов в call_log
        fake_mod = types.ModuleType("dem.gpu_backend_fake_for_test")
        fake_mod.is_available = lambda: True
        def fake_pairwise(particles, cm):
            call_log.append("gpu_pairwise")
            # накапливаем фиктивную силу
            for p in particles:
                p.force[0] += 1.0
                p.force[1] -= 0.5
                p.torque += 0.01
        fake_mod.compute_pairwise_forces_cupy = fake_pairwise
        return fake_mod

    def _particles(self, n):
        import numpy as np
        from dem.particle import Particle
        ps = []
        for i in range(n):
            ps.append(Particle(
                id=i,
                radius=0.02, density=2500.0,
                mass=0.01, inertia=1e-4,
                pos=np.array([0.0, 0.0]),
                vel=np.array([0.0, 0.0]),
                ang_vel=0.0,
                force=np.zeros(2),
                torque=0.0,
                history=[],
            ))
        return ps

    def test_compute_all_forces_uses_gpu_when_enabled(self):
        call_log = []
        fake = self._setup(call_log)
        with patch.dict(sys.modules, {"dem.gpu_backend": fake}):
            from utils.config import SimulationConfig
            from dem.contact_model import ContactModel
            from dem.force_calculation import compute_all_forces
            cfg = SimulationConfig(use_gpu=True, num_particles=5, gravity=0.0)
            cm = ContactModel(kn=1e5, restitution_coeff=0.9,
                             mu_s=0.5, mu_d=0.4,
                             rolling_friction_coeff=0.01,
                             dt=1e-5, config=cfg)
            particles = self._particles(5)
            contacts = compute_all_forces(particles, [], cm)
        self.assertIn("gpu_pairwise", call_log,
                      "compute_all_forces должен вызвать GPU-путь при use_gpu=True")
        # Contacts dict заполняется парными записями (для обратной совместимости):
        # C(5,2) = 10 пар.
        self.assertEqual(len(contacts), 10)
        # Все силы должны быть ненулевыми (фиктивный GPU-bump) с нулевой гравитацией.
        for p in particles:
            self.assertAlmostEqual(p.force[0], 1.0)
            self.assertAlmostEqual(p.force[1], -0.5)
            self.assertAlmostEqual(p.torque, 0.01)

    def test_compute_all_forces_falls_back_without_gpu_flag(self):
        call_log = []
        fake = self._setup(call_log)
        with patch.dict(sys.modules, {"dem.gpu_backend": fake}):
            from utils.config import SimulationConfig
            from dem.contact_model import ContactModel
            from dem.force_calculation import compute_all_forces
            cfg = SimulationConfig(use_gpu=False, gravity=0.0)  # явный CPU
            cm = ContactModel(kn=1e5, restitution_coeff=0.9,
                             mu_s=0.5, mu_d=0.4,
                             rolling_friction_coeff=0.0,
                             dt=1e-5, config=cfg)
            particles = self._particles(5)
            compute_all_forces(particles, [], cm)
        self.assertNotIn("gpu_pairwise", call_log,
                         "если use_gpu=False, GPU-путь вызываться не должен")

    def test_compute_all_forces_falls_back_when_gpu_unavailable(self):
        """Если use_gpu=True, но is_available()=False, должен фолбэчиться на Numba/CPU."""
        call_log = []
        fake = self._setup(call_log)
        fake.is_available = lambda: False  # override
        with patch.dict(sys.modules, {"dem.gpu_backend": fake}):
            from utils.config import SimulationConfig
            from dem.contact_model import ContactModel
            from dem.force_calculation import compute_all_forces
            cfg = SimulationConfig(use_gpu=True, gravity=0.0)
            cm = ContactModel(kn=1e5, restitution_coeff=0.9,
                             mu_s=0.5, mu_d=0.4,
                             rolling_friction_coeff=0.0,
                             dt=1e-5, config=cfg)
            particles = self._particles(5)
            compute_all_forces(particles, [], cm)
        self.assertNotIn("gpu_pairwise", call_log)


if __name__ == "__main__":
    unittest.main()
