from dataclasses import dataclass



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