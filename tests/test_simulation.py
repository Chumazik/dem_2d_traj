import unittest
from dem.simulation import Simulation, SimulationConfig

class TestSimulation(unittest.TestCase):
    def test_simulation_initialization(self):
        config = SimulationConfig(
            num_particles=10,
            particle_radius=0.02,
            particle_density=2500,
            kn=1e5,
            restitution_coeff=0.9,
            friction_static=0.5,
            friction_dynamic=0.4,
            rolling_friction=0.01,
            drum_radius=0.5,
            drum_omega=2.0,
            dt=1e-5,
            total_time=0.1
        )
        simulation = Simulation(config)
        self.assertEqual(len(simulation.particles), 10)
        self.assertEqual(len(simulation.boundaries), 1)
        self.assertIsNotNone(simulation.contact_model)

    def test_initialize_particles(self):
        config = SimulationConfig(
            num_particles=5,
            particle_radius=0.02,
            particle_density=2500,
            kn=1e5,
            restitution_coeff=0.9,
            friction_static=0.5,
            friction_dynamic=0.4,
            rolling_friction=0.01,
            drum_radius=0.5,
            drum_omega=2.0,
            dt=1e-5,
            total_time=0.1
        )
        simulation = Simulation(config)
        self.assertEqual(len(simulation.particles), 5)
        for particle in simulation.particles:
            self.assertEqual(particle.radius, 0.02)
            self.assertEqual(particle.density, 2500)
            self.assertGreater(particle.mass, 0)
            self.assertGreater(particle.inertia, 0)
            self.assertEqual(len(particle.history), 1)

    def test_initialize_boundaries(self):
        config = SimulationConfig(
            num_particles=10,
            particle_radius=0.02,
            particle_density=2500,
            kn=1e5,
            restitution_coeff=0.9,
            friction_static=0.5,
            friction_dynamic=0.4,
            rolling_friction=0.01,
            drum_radius=0.5,
            drum_omega=2.0,
            num_lifters=0,
            lifter_height=0.0,
            dt=1e-5,
            total_time=0.1
        )
        simulation = Simulation(config)
        self.assertEqual(len(simulation.boundaries), 1)
        boundary = simulation.boundaries[0]
        self.assertEqual(boundary.radius, 0.5)
        self.assertEqual(boundary.omega, 2.0)
        self.assertEqual(boundary.applied_torque, 0.0)

    def test_lifters_rotate_when_step_is_called_externally(self):
        """Лифтеры должны вращаться даже если step() вызывается напрямую,
        а не через run(). Иначе физика статична при внешних циклах."""
        from dem.geometry import Lifter
        config = SimulationConfig(
            num_particles=5,
            num_lifters=4,
            lifter_height=0.03,
            lifter_width=0.02,
            drum_radius=0.5,
            drum_omega=2.0,
            dt=1e-5,
            total_time=0.1,
        )
        sim = Simulation(config)
        lifters = [b for b in sim.boundaries if isinstance(b, Lifter)]
        self.assertEqual(len(lifters), 4)
        base0 = lifters[0].base_angle
        # Начальные углы должны быть base_angle.
        for lifter in lifters:
            self.assertAlmostEqual(lifter.current_angle, lifter.base_angle, places=8)
        for _ in range(50):
            sim.step()
        # Угол должен продвинуться вперёд (omega*dt*N).
        # При адаптивном dt используем фактическое время симуляции
        actual_time = sim._sim_time
        expected = base0 + 2.0 * actual_time
        self.assertAlmostEqual(lifters[0].current_angle, expected, places=7)

    def test_max_force_and_velocity_histories_are_recorded_per_step(self):
        """Каждый step() должен добавлять по одному значению max |F| и max |v|
        по всем частицам в соответствующие истории."""
        cfg = SimulationConfig(num_particles=5, gravity=9.81, dt=1e-4)
        sim = Simulation(cfg)
        for _ in range(20):
            sim.step()
        # Истории имеют тот же размер, что и step_count.
        self.assertEqual(len(sim.max_force_history), 20)
        self.assertEqual(len(sim.max_velocity_history), 20)
        # Все значения неотрицательные (нормы).
        self.assertTrue(all(v >= 0 for v in sim.max_force_history))
        self.assertTrue(all(v >= 0 for v in sim.max_velocity_history))
        # Под гравитацией первая частица со временем ускоряется →
        # max |v| должен быть неотрицательным и расти.
        self.assertGreater(sim.max_velocity_history[-1], 0.0)
        # Если частицы находятся в контакте (стенка или лифтеры), max |F| > 0.
        # Свободное падение даст F=m·g для одной частицы.
        self.assertGreater(sim.max_force_history[0], 0.0)

    def test_max_force_reflects_largest_contact_over_all_particles(self):
        """max |F| соответствует наибольшей силе, действующей на одну частицу."""
        import numpy as np
        cfg = SimulationConfig(num_particles=3, gravity=9.81, drum_radius=0.5)
        sim = Simulation(cfg)
        sim.step()
        expected = max(
            float(np.linalg.norm(p.force)) for p in sim.particles
        )
        self.assertAlmostEqual(
            float(sim.max_force_history[0]), expected, places=8
        )

    def test_max_velocity_reflects_largest_speed_over_all_particles(self):
        import numpy as np
        cfg = SimulationConfig(num_particles=4, gravity=9.81, dt=1e-4)
        sim = Simulation(cfg)
        for _ in range(50):
            sim.step()
        expected = max(
            float(np.linalg.norm(p.vel)) for p in sim.particles
        )
        self.assertAlmostEqual(
            float(sim.max_velocity_history[-1]), expected, places=8
        )

if __name__ == '__main__':
    unittest.main()
