import numpy as np
from dataclasses import dataclass, field
from typing import List

@dataclass
class Particle:
    """Класс частицы DEM."""
    id: int
    radius: float
    density: float
    mass: float
    inertia: float                     # I = 0.5 * m * r^2
    pos: np.ndarray                    # позиция (x, y)
    vel: np.ndarray                    # скорость (vx, vy)
    ang_vel: float                     # угловая скорость
    force: np.ndarray = field(default_factory=lambda: np.zeros(2))
    torque: float = 0.0
    history: List[np.ndarray] = field(default_factory=list)

    def apply_force(self, force_vector: np.ndarray, torque_value: float) -> None:
        """Добавляет к текущим суммам силу и крутящий момент."""
        self.force += force_vector
        self.torque += torque_value

    def reset_force(self) -> None:
        """Обнуляет накопленные силы и моменты."""
        self.force = np.zeros(2)
        self.torque = 0.0

    def update_history(self) -> None:
        """Сохраняет копию текущей позиции в истории."""
        self.history.append(self.pos.copy())