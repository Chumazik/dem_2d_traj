import math
import unittest

from dem.analytical import AnalyticalParams, compute_analytical, to_jsonable


class TestAnalytical(unittest.TestCase):

    def test_critical_speed_matches_sheet(self):
        """Критическая скорость обязана совпадать с таблицей Moly-Cop."""
        p = AnalyticalParams(
            effective_mill_diameter_ft=36.0,
            ball_diameter_in=5.0,
            pct_critical_speed=76.0,
        )
        out = compute_analytical(p)
        self.assertAlmostEqual(out.critical_speed_rad_s, 1.34473, places=4)
        self.assertAlmostEqual(out.critical_speed_rpm, 12.8412, places=3)

    def test_operating_speed_is_phi_times_critical(self):
        p = AnalyticalParams(
            effective_mill_diameter_ft=36.0,
            ball_diameter_in=5.0,
            pct_critical_speed=76.0,
        )
        out = compute_analytical(p)
        self.assertAlmostEqual(
            out.operating_speed_rad_s, 0.76 * out.critical_speed_rad_s, places=10
        )
        self.assertAlmostEqual(out.fraction_of_critical, 0.76, places=10)

    def test_classical_shoulder_breakaway(self):
        """Классический угол отрыва shoulder = acos(φ²)."""
        p = AnalyticalParams(pct_critical_speed=76.0)
        out = compute_analytical(p)
        expected = math.acos(0.76 ** 2)
        self.assertAlmostEqual(out.shoulder_angle_rad, expected, places=10)
        self.assertAlmostEqual(out.shoulder_angle_deg, math.degrees(expected), places=8)

    def test_impact_distance_is_shell_radius(self):
        """Точка удара должна лежать на окружности барабана."""
        p = AnalyticalParams()
        out = compute_analytical(p)
        self.assertAlmostEqual(
            out.impact_distance_from_center_ft, out.mill_radius_ft, places=4
        )

    def test_ball_mass_consistency(self):
        p = AnalyticalParams(ball_diameter_in=3.0)
        out = compute_analytical(p)
        expected_lbm = 0.284 * (4.0 / 3.0) * math.pi * (1.5 ** 3)
        self.assertAlmostEqual(out.ball_mass_lbm, expected_lbm, places=6)
        self.assertAlmostEqual(
            out.ball_mass_kg, out.ball_mass_lbm * 0.45359237, places=6
        )

    def test_filling_segment_area_matches_target(self):
        """kidney_angle порождает сегмент, площадь которого пропорциональна fill."""
        p = AnalyticalParams(apparent_mill_filling=28.0)
        out = compute_analytical(p)
        seg = (out.kidney_angle_rad - math.sin(out.kidney_angle_rad)) / (2.0 * math.pi)
        self.assertAlmostEqual(seg, 0.28, places=4)

    def test_trajectory_table_shape_and_boundary(self):
        p = AnalyticalParams()
        out = compute_analytical(p, n_traj_points=11)
        self.assertEqual(len(out.trajectory), 11)
        t0, x0, y0, _ = out.trajectory[0]
        tn, xn, yn, _ = out.trajectory[-1]
        self.assertAlmostEqual(t0, 0.0, places=12)
        self.assertAlmostEqual(x0, out.shoulder_position_x_ft, places=6)
        self.assertAlmostEqual(y0, out.shoulder_position_y_ft, places=6)
        # Конечная точка ≈ точка удара
        self.assertAlmostEqual(xn, out.impact_position_x_ft, places=4)
        self.assertAlmostEqual(yn, out.impact_position_y_ft, places=4)

    def test_to_jsonable_serializes_correctly(self):
        p = AnalyticalParams()
        out = compute_analytical(p)
        data = to_jsonable(out)
        # Таблица траектории сериализуется как список словарей
        self.assertIsInstance(data["trajectory"], list)
        for row in data["trajectory"]:
            self.assertEqual(set(row.keys()), {"t", "x", "y", "v"})
        # Все скалярные поля — числа
        self.assertIsInstance(data["critical_speed_rad_s"], float)
        self.assertIsInstance(data["diff_vs_molycop"], dict)

    def test_trajectory_handles_different_fractions(self):
        for phi in (0.5, 0.7, 0.85):
            p = AnalyticalParams(pct_critical_speed=phi * 100)
            out = compute_analytical(p)
            self.assertAlmostEqual(
                out.fraction_of_critical, phi, places=10
            )
            self.assertGreater(out.impact_speed_ft_s, 0.0)
            self.assertGreater(out.impact_kinetic_energy_joules, 0.0)


class TestTailSnapshot(unittest.TestCase):
    """Перенесён в ``test_live_buffer.py``."""
    pass


if __name__ == "__main__":
    unittest.main()
