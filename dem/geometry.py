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
        contact_point = particle.pos - dist * self.normal
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


class Lifter(Boundary):
    """Прямоугольный лифтер, жестко связанный с вращающимся барабаном."""
    def __init__(self, drum_center, drum_radius, height, width, base_angle, omega):
        self.drum_center = np.array(drum_center, dtype=float)
        self.drum_radius = drum_radius
        self.height = height
        self.width = width
        self.base_angle = base_angle
        self.omega = omega
        self.current_angle = base_angle
        self.applied_torque = 0.0
        
        self._update_corners()

    def _update_corners(self):
        r_out = self.drum_radius
        r_in = self.drum_radius - self.height
        
        tangent = np.array([-np.sin(self.current_angle), np.cos(self.current_angle)])
        
        center_out = self.drum_center + r_out * np.array([np.cos(self.current_angle), np.sin(self.current_angle)])
        center_in = self.drum_center + r_in * np.array([np.cos(self.current_angle), np.sin(self.current_angle)])
        
        self.p1 = center_out - (self.width / 2) * tangent
        self.p2 = center_out + (self.width / 2) * tangent
        self.p3 = center_in + (self.width / 2) * tangent
        self.p4 = center_in - (self.width / 2) * tangent

    def update_time(self, t):
        self.current_angle = self.base_angle + self.omega * t
        self._update_corners()

    def detect_collision(self, particle):
        segments = [
            (self.p1, self.p2),
            (self.p2, self.p3),
            (self.p3, self.p4),
            (self.p4, self.p1)
        ]
        
        min_dist_sq = float('inf')
        closest_pt = None
        
        for p_a, p_b in segments:
            pt = self._closest_point_on_segment(particle.pos, p_a, p_b)
            dist_sq = np.sum((particle.pos - pt)**2)
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_pt = pt
                
        dist = np.sqrt(min_dist_sq)
        inside = self._is_inside(particle.pos)
        
        if not inside and dist >= particle.radius:
            return None
            
        if inside:
            overlap = particle.radius + dist
            if dist > 1e-8:
                normal_unit_vector = (particle.pos - closest_pt) / dist
            else:
                normal_unit_vector = np.array([np.cos(self.current_angle), np.sin(self.current_angle)])
            contact_point = closest_pt
        else:
            overlap = particle.radius - dist
            normal_unit_vector = (particle.pos - closest_pt) / dist
            contact_point = closest_pt
            
        r_vec = contact_point - self.drum_center
        surface_vel = np.array([-self.omega * r_vec[1], self.omega * r_vec[0]])
        
        rel_vel = particle.vel - surface_vel
        overlap_rate = np.dot(rel_vel, normal_unit_vector)
        tangential_velocity = rel_vel - overlap_rate * normal_unit_vector
        
        return overlap, contact_point, normal_unit_vector, overlap_rate, tangential_velocity

    def _closest_point_on_segment(self, p, a, b):
        ab = b - a
        ap = p - a
        ab_dot = np.dot(ab, ab)
        if ab_dot < 1e-12:
            return a
        t = np.dot(ap, ab) / ab_dot
        t = max(0.0, min(1.0, t))
        return a + t * ab
        
    def _is_inside(self, p):
        pts = [self.p1, self.p2, self.p3, self.p4]
        n = len(pts)
        sign = None
        for i in range(n):
            a = pts[i]
            b = pts[(i+1)%n]
            cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
            if sign is None:
                sign = cross > 0
            elif (cross > 0) != sign:
                return False
        return True

    def apply_driving_torque(self, torque):
        """Накопление реактивного момента."""
        self.applied_torque += torque
