import unittest

import numpy as np

from dem.geometry import WallLine, WallCircle

from dem.particle import Particle



class TestGeometry(unittest.TestCase):

    def test_wall_line_detect_collision(self):

        # Create a wall line (vertical line at x=1)

        wall = WallLine(point=[1, 0], normal=[-1, 0])



        # Create a particle

        particle = Particle(

            id=0, radius=0.02, density=2500, mass=0.01, inertia=0.0001,

            pos=np.array([0.9, 0]), vel=np.array([0, 0]), ang_vel=0.0,

            force=np.array([0, 0]), torque=0.0, history=[np.array([0, 0])]

        )



        # Detect collision

        result = wall.detect_collision(particle)



        # Check that collision is detected

        self.assertIsNotNone(result)

        overlap, contact_point, normal_unit_vector, overlap_rate, tangential_velocity = result



        self.assertGreater(overlap, 0)

        self.assertEqual(len(contact_point), 2)

        self.assertEqual(len(normal_unit_vector), 2)

        self.assertIsInstance(overlap_rate, (int, float))

        self.assertEqual(len(tangential_velocity), 2)



    def test_wall_circle_detect_collision(self):

        # Create a drum boundary

        drum = WallCircle(center=[0, 0], radius=0.5, omega=2.0)



        # Create a particle inside the drum

        particle = Particle(

            id=0, radius=0.02, density=2500, mass=0.01, inertia=0.0001,

            pos=np.array([0.4, 0]), vel=np.array([0, 0]), ang_vel=0.0,

            force=np.array([0, 0]), torque=0.0, history=[np.array([0, 0])]

        )



        # Detect collision

        result = drum.detect_collision(particle)



        # Check that collision is detected

        self.assertIsNotNone(result)

        overlap, contact_point, normal_unit_vector, overlap_rate, tangential_velocity = result



        self.assertGreater(overlap, 0)

        self.assertEqual(len(contact_point), 2)

        self.assertEqual(len(normal_unit_vector), 2)

        self.assertIsInstance(overlap_rate, (int, float))

        self.assertEqual(len(tangential_velocity), 2)



    def test_wall_circle_apply_driving_torque(self):

        # Create a drum boundary

        drum = WallCircle(center=[0, 0], radius=0.5, omega=2.0)



        # Apply torque

        torque = 1.0

        drum.apply_driving_torque(torque)



        # Check that torque is accumulated

        self.assertEqual(drum.applied_torque, torque)



        # Apply another torque

        drum.apply_driving_torque(2.0)



        # Check that torques are summed

        self.assertEqual(drum.applied_torque, 3.0)



if __name__ == '__main__':

    unittest.main()