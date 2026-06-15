import numpy as np
from .force_calculation import compute_all_forces


def velocity_verlet_step(particles, dt, contact_model, boundaries):

    # Half-step velocity update

    for particle in particles:

        a = particle.force / particle.mass

        alpha = particle.torque / particle.inertia

        particle.vel += 0.5 * a * dt

        particle.ang_vel += 0.5 * alpha * dt



    # Update positions

    for particle in particles:

        particle.pos += particle.vel * dt

        particle.ang_vel += particle.ang_vel * dt



    # Reset forces and torques

    for particle in particles:

        particle.force = np.zeros(2)

        particle.torque = 0.0



    # Compute all forces

    compute_all_forces(particles, boundaries, contact_model)



    # Second half-step velocity update with new forces

    for particle in particles:

        a = particle.force / particle.mass

        alpha = particle.torque / particle.inertia

        particle.vel += 0.5 * a * dt

        particle.ang_vel += 0.5 * alpha * dt



    # Save history

    for particle in particles:

        particle.update_history()