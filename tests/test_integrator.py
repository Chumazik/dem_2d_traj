import unittest

import numpy as np

from dem.integrator import velocity_verlet_step

from dem.contact_model import ContactModel

from dem.particle import Particle

from dem.geometry import WallCircle



class TestIntegrator(unittest.TestCase):

    def test_velocity_verlet_step(self):

        # Create particles

        particle1 = Particle(

            id=0, radius=0.02, density=2500, mass=0.01, inertia=0.0001,

            pos=np.array([0.0, 0.0]), vel=np.array([1.0, 0.0]), ang_vel=0.0,

            force=np.array([0.0, 0.0]), torque=0.0, history=[np.array([0.0, 0.0])]

        )



        particle2 = Particle(

            id=1, radius=0.02, density=2500, mass=0.01, inertia=0.0001,

            pos=np.array([0.04, 0.0]), vel=np.array([0.0, 0.0]), ang_vel=0.0,

            force=np.array([0.0, 0.0]), torque=0.0, history=[np.array([0.0, 0.0])]

        )



        particles = [particle1, particle2]



        # Create boundary

        boundary = WallCircle(center=[0, 0], radius=0.5, omega=0.0)



        # Create contact model

        contact_model = ContactModel(kn=1e5, restitution_coeff=0.9)



        # Run one step

        dt = 1e-5

        velocity_verlet_step(particles, dt, contact_model, [boundary])



        # Check that the moving particle has moved

        self.assertNotEqual(particle1.pos[0], 0.0)

        # The stationary particle (no forces, no velocity) stays put

        self.assertAlmostEqual(particle2.pos[0], 0.04)



        # Check that history was updated

        self.assertEqual(len(particle1.history), 2)

        self.assertEqual(len(particle2.history), 2)



if __name__ == '__main__':

    unittest.main()