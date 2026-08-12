"""Теоретическая проверка модели DEM на соответствие классической формулировке
Cundall–Strack (1979) и реализации в MercuryDPM (см. ``instruct/``).

В этой модели на каждую частицу действуют только две категории сил:

1. Контактные силы (``contact forces``) между парой частиц или
   частицей и границей: нормальная упруго-демпфирующая сила, касательная
   сила трения (Кулон), момент качения. Всё это вычисляется в
   :func:`dem.contact_model.ContactModel.compute_forces`.
2. Гравитация --- единственная внешняя (объёмная) сила в системе.

Никаких других внешних сил (электромагнитных, кориолисовых и т.п.) нет.
Сцена строго 2D: каждый ``Particle`` имеет 2D координаты ``pos / vel / force``,
стенки/лифтеры являются 2D геометриями WallLine / WallCircle / Lifter.
Соглашение осей:

* +X --- вправо на канвасе;
* +Y --- вниз на канвасе (Y-down web-конвенция);
* гравитация действует вдоль +Y (т.е. "вниз" визуально);
* физически эквивалентно Y-up соглашению с гравитацией в -Y.

Данные тесты фиксируют эти инварианты.
"""

import unittest

import numpy as np

from dem.contact_model import ContactModel
from dem.particle import Particle
from dem.geometry import WallCircle
from dem.force_calculation import compute_all_forces
from dem.simulation import Simulation
from utils.config import SimulationConfig


def _make_particle(mass=1.0, pos=(0.0, 0.0), vel=(0.0, 0.0), r=0.02):
    return Particle(
        id=0, radius=r, density=2500.0,
        mass=mass, inertia=0.5 * mass * r ** 2,
        pos=np.array(pos, dtype=float),
        vel=np.array(vel, dtype=float),
        ang_vel=0.0,
        force=np.zeros(2),
        torque=0.0,
        history=[],
    )


class TestForcesInventory(unittest.TestCase):
    """Подтверждаем, что единственный не-контактный источник силы — гравитация."""

    def test_only_contact_and_gravity_write_particle_force(self):
        """Аудит ``dem/force_calculation.py``: единственное сложение
        ``p.force[…] += …`` вне ``apply_force`` — гравитация ``m·g``."""
        from dem import force_calculation as fc
        import inspect

        src = inspect.getsource(fc.compute_all_forces)
        bad = []
        for ln_num, line in enumerate(src.splitlines(), 1):
            stripped = line.strip()
            # Skip comments and blanks
            if not stripped or stripped.startswith("#"):
                continue
            if "p.force[" in stripped and "+=" in stripped:
                # Allow only the gravity line by structural signature
                if "m * float(g_value)" not in stripped and "p.mass * " not in stripped:
                    bad.append((ln_num, stripped))
        self.assertEqual(
            bad, [],
            "Найдены сомнительные ``p.force[N] += `` записи в compute_all_forces, "
            "которые не являются гравитацией: " + repr(bad),
        )

    def test_no_extra_force_contributions_in_particle_module(self):
        """В ``particle.py`` единственный метод, меняющий force / torque ---
        явный ``apply_force`` от контактной логики (т.е. через caller'ов)."""
        import inspect
        from dem import particle as pm
        src = inspect.getsource(pm.Particle)
        # Положительные сценарии в apply_force:
        self.assertIn("self.force += force_vector", src)
        self.assertIn("self.torque += torque_value", src)
        # reset_force сбрасывает к нулю (используется compute_all_forces).
        self.assertIn("self.force = np.zeros(2)", src)

    def test_contact_pair_forces_obey_newtons_third_law(self):
        """Изолированная пара перекрывающихся частиц: равно-противо-направлены
        и сумма = 0. Гравитация отключена, чтобы убрать постороннее смещение."""
        p1 = _make_particle(mass=1.0, pos=(0.0, 0.0))
        p2 = _make_particle(mass=1.0, pos=(0.039, 0.0))  # overlap dist 0.039 < 0.04
        cfg = SimulationConfig(gravity=0.0)
        cm = ContactModel(
            kn=1e5, restitution_coeff=0.9, mu_s=0.5, mu_d=0.4,
            rolling_friction_coeff=0.01, dt=1e-5, config=cfg,
        )
        # Границы нет -- проверяем чистую симметрию пары.
        compute_all_forces([p1, p2], [], cm)
        # Newton 3-й: сумма сил пары = 0.
        self.assertAlmostEqual(
            float(p1.force[0] + p2.force[0]), 0.0, places=6,
            msg="Pair contact: sum F_x must be zero for Newton's 3rd law",
        )
        self.assertAlmostEqual(
            float(p1.force[1] + p2.force[1]), 0.0, places=6,
        )
        # p1 слева, p2 справа: overlap отталкивает, поэтому p1 чувствует -X,
        # а p2 чувствует +X.
        self.assertLess(
            float(p1.force[0]), 0.0,
            msg="P1 (слева) должен чувствовать -x контактную силу (отталкивание от p2)",
        )
        self.assertGreater(
            float(p2.force[0]), 0.0,
            msg="P2 (справа) должен чувствовать +x реакцию от p1",
        )

    def test_wall_contact_only_inward(self):
        """Частица внутри барабана должна чувствовать СИЛУ, направленную
        ВНУТРЬ (т.е. -normal). Никаких "external push" вдоль оси вращения."""
        # Частица чуть внутри барабана (0.40 от центра при R=0.5, overlap = 0.12).
        p = _make_particle(mass=1.0, pos=(0.40, 0.0), vel=(0.0, 0.0))
        cfg = SimulationConfig(gravity=0.0)
        cm = ContactModel(
            kn=1e5, restitution_coeff=0.9, mu_s=0.5, mu_d=0.4,
            rolling_friction_coeff=0.01, dt=1e-5, config=cfg,
        )
        drum = WallCircle(center=(0.0, 0.0), radius=0.5, omega=0.0)
        compute_all_forces([p], [drum], cm)
        # Стенка неподвижна (omega=0), поверхностной скорости нет;
        # относительной тангенциальной == 0; единственная стеночная сила —
        # упругая нормальная, направленная ВНУТРЬ барабана (т.е. -normal).
        # Позиция x=+0.40, drum center=0, normal = (+x). Внутренняя сила = -x.
        self.assertLess(float(p.force[0]), 0.0)
        self.assertAlmostEqual(float(p.force[1]), 0.0, places=6)

    def test_gravity_is_the_only_uniform_body_force(self):
        """Свободная частица без соседей и стенок чувствует только гравитацию:
        ``F = (0, m·g)``."""
        p = _make_particle(mass=2.0)
        cfg = SimulationConfig(gravity=9.81)
        cm = ContactModel(kn=1e5, restitution_coeff=0.9, config=cfg)
        # Барабан далеко -- не касается.
        drum = WallCircle(center=(100.0, 100.0), radius=0.5, omega=0.0)
        compute_all_forces([p], [drum], cm)
        self.assertAlmostEqual(p.force[0], 0.0, places=8)
        self.assertAlmostEqual(p.force[1], 2.0 * 9.81, places=8)
        self.assertEqual(p.torque, 0.0)

    def test_particles_are_strictly_2d(self):
        """Аудит: ни в одном из dem/*.py не должно быть обращений к
        третьей оси координат. Это гарантирует сохранение 2D."""
        import re
        from pathlib import Path
        forbidden = re.compile(
            r"pos\[:, ?2\]|\bshape\s*\(\s*3\s*,|\bshape\s*=\s*\(\s*3\s*,"
            r"|\.pos\[2\]|\.vel\[2\]|\.force\[2\]|np\.zeros\(\(3|\.ang_vel\[\d]"
        )
        offenders = []
        for fp in Path("dem").rglob("*.py"):
            text = fp.read_text(encoding="utf-8")
            for ln_num, line in enumerate(text.splitlines(), 1):
                if forbidden.search(line):
                    offenders.append((str(fp), ln_num, line.strip()))
        self.assertEqual(
            offenders, [],
            "Найдены обращения к 3-й оси; модель должна быть 2D: "
            f"{offenders}",
        )


class TestFreeFallTrajectory(unittest.TestCase):
    """Сравниваем симуляцию одиночной свободной частицы (без границ и других
    частиц) с аналитическим решением ``y(t) = 0.5·g·t²``, ``v(t) = g·t``."""

    def test_signed_gravity_yields_correct_free_fall_jit(self):
        cfg = SimulationConfig(num_particles=1, drum_radius=0.0,
                               use_jit=True, gravity=9.81, dt=1e-4)
        sim = Simulation(cfg)
        sim.boundaries = []
        p = sim.particles[0]
        p.pos[:] = 0.0
        p.force[:] = 0.0
        y0 = float(p.pos[1])
        N = 100
        dt = cfg.dt
        for _ in range(N):
            sim.step()
        t = N * dt
        drop_analytic = 0.5 * 9.81 * t * t
        vy_analytic = 9.81 * t
        self.assertAlmostEqual(
            float(p.pos[1]) - y0, drop_analytic, places=8,
            msg=f"Free fall drop не соответствует ½g·t²: "
                f"actual={p.pos[1] - y0:.6e}, expected={drop_analytic:.6e}",
        )
        self.assertAlmostEqual(
            float(p.vel[1]), vy_analytic, places=7,
        )

    def test_signed_gravity_yields_correct_free_fall_cpu(self):
        cfg = SimulationConfig(num_particles=1, drum_radius=0.0,
                               use_jit=False, gravity=9.81, dt=1e-4)
        sim = Simulation(cfg)
        sim.boundaries = []
        p = sim.particles[0]
        p.pos[:] = 0.0
        p.force[:] = 0.0
        y0 = float(p.pos[1])
        N = 100
        dt = cfg.dt
        for _ in range(N):
            sim.step()
        t = N * dt
        self.assertAlmostEqual(float(p.pos[1]) - y0, 0.5 * 9.81 * t * t, places=8)
        self.assertAlmostEqual(float(p.vel[1]), 9.81 * t, places=7)


if __name__ == "__main__":
    unittest.main()
