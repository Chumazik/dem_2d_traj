import numpy as np

from dataclasses import dataclass



@dataclass

class Particle:

    id: int

    radius: float

    density: float

    mass: float

    inertia: float  # I = 0.5 * m * r^2

    pos: np.array  # position (x, y)

    vel: np.array  # velocity (vx, vy)

    ang_vel: float  # angular velocity

    force: np.array  # force vector (fx, fy)

    torque: float  # torque value

    history: list  # list of positions for trajectory



    def apply_force(self, force_vector, torque_value):

        """Add force and torque to the particle."""

        self.force += force_vector

        self.torque += torque_value



    def update_history(self):

        """Update the position history of the particle."""

        self.history.append(np.copy(self.pos))