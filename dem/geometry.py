import numpy as np



class Boundary:

    def detect_collision(self, particle):

        raise NotImplementedError("Subclasses should implement this method.")



class WallLine(Boundary):

    def __init__(self, point, normal):

        self.point = np.array(point)

        self.normal = np.array(normal) / np.linalg.norm(normal)



    def detect_collision(self, particle):

        # Calculate distance from particle center to the line

        dist = np.dot(particle.pos - self.point, self.normal)

        if dist > 0:

            return None



        overlap = particle.radius + dist

        contact_point = particle.pos - overlap * self.normal

        normal_unit_vector = -self.normal

        overlap_rate = np.dot(particle.vel, self.normal)

        tangential_velocity = particle.vel - (np.dot(particle.vel, self.normal) * self.normal)



        return overlap, contact_point, normal_unit_vector, overlap_rate, tangential_velocity



class WallCircle(Boundary):

    def __init__(self, center, radius, omega=0.0):

        self.center = np.array(center)

        self.radius = radius

        self.omega = omega

        self.applied_torque = 0.0



    def detect_collision(self, particle):

        dist_to_center = np.linalg.norm(particle.pos - self.center)

        if dist_to_center > self.radius + particle.radius:

            return None



        overlap = (self.radius + particle.radius) - dist_to_center

        contact_point = self.center + (particle.radius / dist_to_center) * (particle.pos - self.center)

        normal_unit_vector = (contact_point - particle.pos) / np.linalg.norm(contact_point - particle.pos)

        overlap_rate = np.dot(particle.vel, normal_unit_vector) - self.omega * np.cross(normal_unit_vector, particle.pos - self.center)

        tangential_velocity = particle.vel - (np.dot(particle.vel, normal_unit_vector) * normal_unit_vector)



        return overlap, contact_point, normal_unit_vector, overlap_rate, tangential_velocity



    def apply_driving_torque(self, torque):

        """Apply reactive torque from particles to the drum."""

        self.applied_torque += torque