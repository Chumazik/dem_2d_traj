import unittest
import numpy as np
from dem.simulation import Simulation
from utils.config import SimulationConfig

class TestSimulation(unittest.TestCase):

    def test_simulation_initialization(self):
        # Create config
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

        # Create simulation
        simulation = Simulation(config)

        # Check that particles were initialized
        self.assertEqual(len(simulation.particles), 10)

        # Check that boundaries were initialized
        self.assertEqual(len(simulation.boundaries), 1)
        self.assertTrue(isinstance(simulation.boundaries[0], type(simulation.boundaries[0])))

        # Check that contact model was created
        self.assertIsNotNone(simulation.contact_model)

    def test_initialize_particles(self):
        # Create config with small number of particles
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

        # Create simulation
        simulation = Simulation(config)

        # Check that particles were initialized correctly
        self.assertEqual(len(simulation.particles), 5)

        for particle in simulation.particles:
            self.assertEqual(particle.radius, 0.02)
            self.assertEqual(particle.density, 2500)
            self.assertGreater(particle.mass, 0)
            self.assertGreater(particle.inertia, 0)
            self.assertEqual(len(particle.history), 1)  # Should have initial position

    def test_initialize_boundaries(self):
        # Create config
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

        # Create simulation
        simulation = Simulation(config)

        # Check that boundaries were initialized correctly
        self.assertEqual(len(simulation.boundaries), 1)
        boundary = simulation.boundaries[0]
        self.assertEqual(boundary.radius, 0.5)
        self.assertEqual(boundary.omega, 2.0)
        self.assertEqual(boundary.applied_torque, 0.0)

if __name__ == '__main__':
    unittest.main()
