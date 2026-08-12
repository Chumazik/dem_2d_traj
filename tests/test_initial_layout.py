import math
import unittest

import numpy as np

from dem.simulation import Simulation
from utils.config import SimulationConfig


class TestInitialParticleLayout(unittest.TestCase):
    """Частицы в начале расчёта должны быть сгенерированы:

    * с небольшим зазором между центрами;
    * согласованно с окном предпросмотра (осевшая куча на дне барабана);
    * с нулевой начальной скоростью."""

    def _layout_is_settled_heap(self, sim, expected_gap_frac):
        d = 2 * sim.config.particle_radius
        spacing = d * (1 + expected_gap_frac)
        # В шахматной (гексагональной) упаковке есть два типа соседств:
        #  - внутри ряда: centers на расстоянии ``spacing`` друг от друга;
        #  - между чётным и нечётным рядом: расстояние sqrt(3)/2 * spacing.
        # Это и есть минимальная допустимая дистанция.
        min_dist = spacing * math.sqrt(3) / 2
        for i, p in enumerate(sim.particles):
            r = np.hypot(p.pos[0], p.pos[1])
            self.assertLessEqual(
                r,
                sim.config.drum_radius - sim.config.particle_radius + 1e-9,
                f"частица[{i}] вышла за эффективную область: r={r:.4f}",
            )
            for j in range(i + 1, len(sim.particles)):
                d2 = np.linalg.norm(p.pos - sim.particles[j].pos)
                self.assertGreaterEqual(
                    d2, min_dist - 1e-9,
                    f"частицы[{i}] и [{j}] слишком близко: dist={d2:.6f} < {min_dist:.6f}",
                )

    def test_zero_initial_velocity(self):
        cfg = SimulationConfig(num_particles=20)
        sim = Simulation(cfg)
        for p in sim.particles:
            self.assertEqual(float(p.vel[0]), 0.0)
            self.assertEqual(float(p.vel[1]), 0.0)
            self.assertEqual(float(p.ang_vel), 0.0)

    def test_all_particles_inside_effective_drum(self):
        cfg = SimulationConfig(num_particles=20, drum_radius=0.5,
                               particle_radius=0.02, gap_fraction=0.05)
        sim = Simulation(cfg)
        eff_R = cfg.drum_radius - cfg.particle_radius
        for i, p in enumerate(sim.particles):
            r = np.hypot(p.pos[0], p.pos[1])
            self.assertLessEqual(
                r, eff_R + 1e-9,
                f"частица[{i}] вышла за эффективную область: r={r:.4f}",
            )

    def test_particles_have_visible_gap(self):
        cfg = SimulationConfig(num_particles=30, drum_radius=0.5,
                               particle_radius=0.02, gap_fraction=0.05)
        sim = Simulation(cfg)
        self._layout_is_settled_heap(sim, expected_gap_frac=0.05)

    def test_particles_match_preview_layout(self):
        """Алгоритм упаковки в Python и в JS-превью совпадают по форме."""
        cfg = SimulationConfig(num_particles=60, drum_radius=0.5,
                               particle_radius=0.02, gap_fraction=0.05,
                               apparent_mill_filling=35.0,
                               angle_of_repose_deg=33.0)
        sim = Simulation(cfg)
        self.assertEqual(len(sim.particles), 60)

        effective_R = cfg.drum_radius - cfg.particle_radius
        d = 2 * cfg.particle_radius
        spacing = d * (1 + cfg.gap_fraction)
        row_h = spacing * math.sqrt(3) / 2
        repose_rad = math.radians(cfg.angle_of_repose_deg)
        bed_radius = effective_R * (0.4 + 0.5 *
                                  math.sqrt(cfg.apparent_mill_filling / 100.0 / 0.5))
        safe_ratio = math.sin(repose_rad) / max(math.cos(repose_rad), 1e-6)

        # Эталонная раскладка по тому же алгоритму (псевдо-python).
        expected = []
        row = 0
        while True:
            y_base = effective_R - row_h * row - cfg.particle_radius
            if y_base < -effective_R + cfg.particle_radius:
                break
            y_above = (effective_R - y_base) / effective_R
            half_w = bed_radius * max(0.0, 1.0 - y_above * safe_ratio)
            row_offset = spacing / 2.0 if row % 2 else 0.0
            x = -half_w - row_offset
            while x <= half_w + 1e-9 and len(expected) < cfg.num_particles:
                if x * x + y_base * y_base <= effective_R * effective_R:
                    expected.append((row, round(x, 6), round(y_base, 6)))
                x += spacing
            row += 1

        # Каждая частица в симуляции должна совпадать с ожидаемой позицией
        # (с точностью до округления).
        actual = sorted([(round(p.pos[1], 6), round(p.pos[0], 6)) for p in sim.particles])
        exp = sorted([(round(y, 6), round(x, 6)) for (_, x, y) in expected])
        self.assertTrue(
            actual == exp,
            f"layouts differ: actual[:5]={actual[:5]} expected[:5]={exp[:5]}",
        )

    def test_layout_with_falling_default_params(self):
        """С дефолтными параметрами (100 частиц, R=0.5, r=0.02) куча собирается
        и ни одна частица не выходит за эффективную область."""
        cfg = SimulationConfig(num_particles=100)
        sim = Simulation(cfg)
        self.assertEqual(len(sim.particles), 100)
        eff_R = cfg.drum_radius - cfg.particle_radius
        all_packed_in_drum = all(
            np.hypot(p.pos[0], p.pos[1]) <= eff_R + 1e-9 for p in sim.particles
        )
        self.assertTrue(all_packed_in_drum)

    def test_gap_fraction_changes_spacing(self):
        d = 2 * 0.02
        for gap_frac in (0.0, 0.05, 0.20):
            cfg = SimulationConfig(num_particles=20, gap_fraction=gap_frac)
            sim = Simulation(cfg)
            min_d = None
            for i, p in enumerate(sim.particles):
                for q in sim.particles[i + 1:]:
                    dd = np.linalg.norm(p.pos - q.pos)
                    if min_d is None or dd < min_d:
                        min_d = dd
            # Минимальная дистанция в шахматной упаковке — между соседними
            # рядами; она равна sqrt(3)/2 * spacing.
            expected_min = d * (1.0 + gap_frac) * math.sqrt(3) / 2
            self.assertGreaterEqual(
                min_d, expected_min - 1e-6,
                f"gap_fraction={gap_frac}: min dist={min_d:.6f} < {expected_min:.6f}",
            )


if __name__ == "__main__":
    unittest.main()
