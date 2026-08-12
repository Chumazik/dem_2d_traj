import unittest

import numpy as np

from dem.geometry import WallLine, WallCircle

from dem.particle import Particle



class TestGeometry(unittest.TestCase):

    def test_wall_line_detect_collision(self):

        # Create a wall line (vertical line at x=1, normal points into the region)

        wall = WallLine(point=[1, 0], normal=[-1, 0])



        # Create a particle whose center has crossed the wall plane

        particle = Particle(

            id=0, radius=0.02, density=2500, mass=0.01, inertia=0.0001,

            pos=np.array([1.02, 0]), vel=np.array([0, 0]), ang_vel=0.0,

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

        # Create a drum boundary (radius 0.5, particle radius 0.02)

        drum = WallCircle(center=[0, 0], radius=0.5, omega=2.0)



        # Create a particle whose surface has reached/passed the inner wall:
        # center at dist = 0.49 > R - r = 0.48 -> contact, overlap = 0.49+0.02-0.5 = 0.01.

        particle = Particle(

            id=0, radius=0.02, density=2500, mass=0.01, inertia=0.0001,

            pos=np.array([0.49, 0]), vel=np.array([0, 0]), ang_vel=0.0,

            force=np.array([0, 0]), torque=0.0, history=[np.array([0, 0])]

        )



        # Detect collision

        result = drum.detect_collision(particle)



        # Check that collision is detected

        self.assertIsNotNone(result)

        overlap, contact_point, normal_unit_vector, overlap_rate, tangential_velocity = result



        self.assertAlmostEqual(overlap, 0.01, places=6)

        self.assertEqual(len(contact_point), 2)

        self.assertEqual(len(normal_unit_vector), 2)

        self.assertIsInstance(overlap_rate, (int, float))

        self.assertEqual(len(tangential_velocity), 2)

        # Нормаль направлена наружу от центра барабана.

        self.assertAlmostEqual(normal_unit_vector[0], 1.0, places=8)

        self.assertAlmostEqual(normal_unit_vector[1], 0.0, places=8)

        # Точка контакта — на поверхности барабана (на расстоянии R от центра).

        self.assertAlmostEqual(np.hypot(contact_point[0], contact_point[1]), 0.5, places=6)

    def test_wall_circle_no_contact_when_inside(self):

        """Частица полностью внутри барабана (dist <= R - r) НЕ должна давать контакта."""

        drum = WallCircle(center=[0, 0], radius=0.5, omega=2.0)

        particle = Particle(

            id=0, radius=0.02, density=2500, mass=0.01, inertia=0.0001,

            pos=np.array([0.3, 0]), vel=np.array([0, 0]), ang_vel=0.0,

            force=np.array([0, 0]), torque=0.0, history=[np.array([0, 0])]

        )

        self.assertIsNone(drum.detect_collision(particle))



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