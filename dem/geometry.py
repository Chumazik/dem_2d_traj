import numpy as np

class Boundary:
    """Базовый класс границы."""
    def detect_collision(self, particle):
        raise NotImplementedError("Subclasses must implement detect_collision.")

class WallLine(Boundary):
    """Прямая стенка, ограничивающая область с внутренней стороны против нормали."""
    def __init__(self, point, normal):
        self.point = np.array(point, dtype=float)
        self.normal = np.array(normal, dtype=float)
        self.normal /= np.linalg.norm(self.normal)

    def detect_collision(self, particle):
        # Расстояние от центра частицы до линии (положительно – в сторону нормали)
        dist = np.dot(particle.pos - self.point, self.normal)

        if dist > 0:
            return None  # нет контакта

        overlap = particle.radius - dist
        contact_point = particle.pos - overlap * self.normal
        normal_unit_vector = -self.normal
        overlap_rate = np.dot(particle.vel, self.normal)
        tangential_velocity = particle.vel - overlap_rate * self.normal

        return overlap, contact_point, normal_unit_vector, overlap_rate, tangential_velocity

class WallCircle(Boundary):
    """Вращающийся барабан (окружность)."""
    def __init__(self, center, radius, omega=0.0):
        self.center = np.array(center, dtype=float)
        self.radius = radius
        self.omega = omega                # угловая скорость барабана
        self.applied_torque = 0.0         # реактивный момент от частиц

    def detect_collision(self, particle):
        # Вектор от центра барабана к центру частицы
        vec = particle.pos - self.center
        dist = np.linalg.norm(vec)

        if dist > self.radius + particle.radius:
            return None  # нет контакта

        overlap = self.radius + particle.radius - dist
        if dist == 0.0:
            # Если частица точно в центре, выбираем произвольное направление
            normal = np.array([1.0, 0.0])
            contact_point = self.center + self.radius * normal
        else:
            normal = vec / dist
            contact_point = self.center + (self.radius / dist) * vec

        # Скорость поверхности барабана в точке контакта
        surface_vel = np.array([-self.omega * (contact_point[1] - self.center[1]),
                                self.omega * (contact_point[0] - self.center[0])])

        rel_vel = particle.vel - surface_vel
        overlap_rate = np.dot(rel_vel, normal)
        tangential_velocity = rel_vel - overlap_rate * normal

        return overlap, contact_point, normal, overlap_rate, tangential_velocity

    def apply_driving_torque(self, torque):
        """Накопление реактивного момента, который необходимо компенсировать."""
        self.applied_torque += torque
