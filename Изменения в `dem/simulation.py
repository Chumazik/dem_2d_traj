from dataclasses import dataclass
import numpy as np
from .contact_model import ContactModel
from .particle import Particle
from .geometry import WallCircle
from .integrator import velocity_verlet_step

@dataclass
class SimulationConfig:
    num_particles: int = 100
    particle_radius: float = 0.02
    particle_density: float = 2500
    kn: float = 1e5
    restitution: float = 0.9
    friction_static: float = 0.5
    friction_dynamic: float = 0.4
    rolling_friction: float = 0.01
    drum_radius: float = 0.5
    drum_omega: float = 2.0
    dt: float = 1e-5
    total_time: float = 5.0

class Simulation:
    def __init__(self, config):
        self.config = config
        self.particles = []
        self.boundaries = []
        self.contact_model = ContactModel(config.kn, restitution_coeff=config.restitution,
                                         mu_s=config.friction_static, mu_d=config.friction_dynamic,
                                         rolling_friction_coeff=config.rolling_friction)
        self.initialize_particles()
        self.initialize_boundaries()

    def initialize_particles(self):
        for i in range(self.config.num_particles):
            angle = 2 * np.pi * i / self.config.num_particles
            x = self.config.drum_radius * np.cos(angle)
            y = self.config.drum_radius * np.sin(angle)
            pos = np.array([x, y])
            mass = self.config.particle_density * np.pi * (self.config.particle_radius ** 2)
            inertia = 0.5 * mass * (self.config.particle_radius ** 2)
            particle = Particle(id=i, radius=self.config.particle_radius,
                               density=self.config.particle_density,
                               mass=mass, inertia=inertia, pos=pos, vel=np.zeros(2), ang_vel=0.0,
                               force=np.zeros(2), torque=0.0, history=[pos])
            self.particles.append(particle)

    def initialize_boundaries(self):
        self.boundaries.append(WallCircle(center=(0, 0), radius=self.config.drum_radius, omega=self.config.drum_omega))

    def step(self):
        velocity_verlet_step(self.particles, self.config.dt, self.contact_model, self.boundaries)
        if len(self.boundaries) > 0:
            self.torque_history.append(-self.boundaries[0].applied_torque)

    def run(self):
        self.time = []
        self.torque_history = []
        t = 0.0
        while t < self.config.total_time:
            self.step()
            self.time.append(t)
            t += self.config.dt

    def get_trajectories(self):
        return [particle.history for particle in self.particles]
