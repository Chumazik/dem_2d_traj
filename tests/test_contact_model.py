import unittest

import numpy as np

from dem.contact_model import ContactModel

from dem.particle import Particle



class TestContactModel(unittest.TestCase):

    def setUp(self):

        self.contact_model = ContactModel(

            kn=1e5,

            restitution_coeff=0.9,

            mu_s=0.5,

            mu_d=0.4,

            rolling_friction_coeff=0.01

        )



    def test_compute_forces(self):

        # Test with two particles

        particle1 = Particle(

            id=0, radius=0.02, density=2500, mass=0.01, inertia=0.0001,

            pos=np.array([0, 0]), vel=np.array([0, 0]), ang_vel=0.0,

            force=np.array([0, 0]), torque=0.0, history=[np.array([0, 0])]

        )



        particle2 = Particle(

            id=1, radius=0.02, density=2500, mass=0.01, inertia=0.0001,

            pos=np.array([0.03, 0]), vel=np.array([0, 0]), ang_vel=0.0,

            force=np.array([0, 0]), torque=0.0, history=[np.array([0, 0])]

        )



        # Test overlap

        overlap = 0.01

        overlap_rate = 0.0

        tangential_displacement = 0.0

        rel_vel_tang = 0.0

        effective_radius = 0.02

        normal_unit_vector = np.array([1, 0])



        normal_force, tangential_force, torque1, torque2 = self.contact_model.compute_forces(

            overlap, overlap_rate, tangential_displacement, rel_vel_tang,

            effective_radius, normal_unit_vector, particle1, particle2

        )



        # Check that forces are computed correctly

        self.assertIsInstance(normal_force, np.ndarray)

        self.assertIsInstance(tangential_force, np.ndarray)

        self.assertIsInstance(torque1, (int, float))

        self.assertIsInstance(torque2, (int, float))



    def test_compute_forces_single_particle(self):

        # Test with single particle and boundary

        particle = Particle(

            id=0, radius=0.02, density=2500, mass=0.01, inertia=0.0001,

            pos=np.array([0, 0]), vel=np.array([0, 0]), ang_vel=0.0,

            force=np.array([0, 0]), torque=0.0, history=[np.array([0, 0])]

        )



        overlap = 0.01

        overlap_rate = 0.0

        tangential_displacement = 0.0

        rel_vel_tang = 0.0

        effective_radius = 0.02

        normal_unit_vector = np.array([1, 0])



        normal_force, tangential_force, torque1, torque2 = self.contact_model.compute_forces(

            overlap, overlap_rate, tangential_displacement, rel_vel_tang,

            effective_radius, normal_unit_vector, particle

        )



        # Check that forces are computed correctly for single particle

        self.assertIsInstance(normal_force, np.ndarray)

        self.assertIsInstance(tangential_force, np.ndarray)

        self.assertIsInstance(torque1, (int, float))

        self.assertEqual(torque2, 0.0)



if __name__ == '__main__':

    unittest.main()