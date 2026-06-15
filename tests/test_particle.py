import unittest

import numpy as np

from dem.particle import Particle



class TestParticle(unittest.TestCase):

    def test_particle_creation(self):

        # Create a particle

        particle = Particle(

            id=0, radius=0.02, density=2500, mass=0.01, inertia=0.0001,

            pos=np.array([1, 2]), vel=np.array([0.1, 0.2]), ang_vel=0.5,

            force=np.array([0.01, 0.02]), torque=0.001, history=[np.array([1, 2])]

        )



        # Check that all attributes are set correctly

        self.assertEqual(particle.id, 0)

        self.assertEqual(particle.radius, 0.02)

        self.assertEqual(particle.density, 2500)

        self.assertEqual(particle.mass, 0.01)

        self.assertEqual(particle.inertia, 0.0001)

        self.assertTrue(np.array_equal(particle.pos, np.array([1, 2])))

        self.assertTrue(np.array_equal(particle.vel, np.array([0.1, 0.2])))

        self.assertEqual(particle.ang_vel, 0.5)

        self.assertTrue(np.array_equal(particle.force, np.array([0.01, 0.02])))

        self.assertEqual(particle.torque, 0.001)

        self.assertEqual(len(particle.history), 1)



    def test_apply_force(self):

        # Create a particle

        particle = Particle(

            id=0, radius=0.02, density=2500, mass=0.01, inertia=0.0001,

            pos=np.array([1, 2]), vel=np.array([0.1, 0.2]), ang_vel=0.5,

            force=np.array([0.01, 0.02]), torque=0.001, history=[np.array([1, 2])]

        )



        # Apply force

        force_vector = np.array([0.03, 0.04])

        torque_value = 0.002



        particle.apply_force(force_vector, torque_value)



        # Check that force and torque are updated

        self.assertTrue(np.array_equal(particle.force, np.array([0.04, 0.06])))

        self.assertEqual(particle.torque, 0.003)



    def test_update_history(self):

        # Create a particle

        particle = Particle(

            id=0, radius=0.02, density=2500, mass=0.01, inertia=0.0001,

            pos=np.array([1, 2]), vel=np.array([0.1, 0.2]), ang_vel=0.5,

            force=np.array([0.01, 0.02]), torque=0.001, history=[np.array([1, 2])]

        )



        # Update history

        particle.update_history()



        # Check that history was updated

        self.assertEqual(len(particle.history), 2)

        self.assertTrue(np.array_equal(particle.history[1], np.array([1, 2])))



if __name__ == '__main__':

    unittest.main()