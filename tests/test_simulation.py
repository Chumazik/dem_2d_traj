import unittest
from dem.simulation import Simulation, SimulationConfig

class TestSimulation(unittest.TestCase):
    def test_simulation_initialization(self):
        config = SimulationConfig(
            num_particles=10,
            particle_radius=0.02,
            particle_density=2500,
            kn=1e5,
            restitution=0.9,
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
            restitution=0.9,
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
            restitution=0.9,
            friction_static=0.5,
            friction_dynamic=0.4,
            rolling_friction=0.01,
            drum_radius=0.5,
            drum_omega=2.0,
            dt=1e-5,
            total_time=0.1
        )
        simulation = Simulation(config)
        self.assertEqual(len(simulation.boundaries), 1)
        boundary = simulation.boundaries[0]
        self.assertEqual(boundary.radius, 0.5)
        self.assertEqual(boundary.omega, 2.0)
        self.assertEqual(boundary.applied_torque, 0.0)

if __name__ == '__main__':
    unittest.main()
