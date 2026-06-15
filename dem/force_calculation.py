import numpy as np
from .geometry import WallCircle

def compute_all_forces (particles, boundaries, contact_model):


    num_particles = len(particles)



    for i in range(num_particles):



        for j in range(i + 1, num_particles):



            dist = np.linalg.norm(particles[i].pos - particles[j].pos)



            if dist < particles[i].radius + particles[j].radius:



                overlap = (particles[i].radius + particles[j].radius) - dist



                normal_unit_vector = (particles[j].pos - particles[i].pos) / dist



                rel_vel_tang = np.cross(normal_unit_vector, particles[i].vel - particles[j].vel)



                effective_radius = 0.5 * (particles[i].radius + particles[j].radius)


                normal_force, tangential_force, torque1, torque2 = contact_model.compute_forces(overlap, 0, 0, rel_vel_tang, effective_radius, normal_unit_vector, particles[i], particles[j])

                particles[i].apply_force(-normal_force, -torque1)

                particles[j].apply_force(normal_force, torque2)


    for boundary in boundaries:



        for particle in particles:



            collision_info = boundary.detect_collision(particle)



            if collision_info is not None:



                overlap, contact_point, normal_unit_vector, overlap_rate, tangential_velocity = collision_info



                effective_radius = particle.radius


                normal_force, tangential_force, torque1, _ = contact_model.compute_forces(overlap, overlap_rate, 0, tangential_velocity[1], effective_radius, normal_unit_vector, particle)


                particle.apply_force(-normal_force, -torque1)


                if isinstance(boundary, WallCircle):

                    boundary.apply_driving_torque(torque1)