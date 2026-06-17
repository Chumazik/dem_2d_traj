import numpy as np
from .contact_model import ContactModel

def compute_all_forces(particles, boundaries, contact_model: ContactModel):
    """
    Вычисляет все взаимодействия частиц‑частиц и частица‑граница.
    Сохраняет реактивный момент в объекте WallCircle.
    """
    n = len(particles)

    # ---- Частица‑частица ----
    for i in range(n):
        for j in range(i + 1, n):
            pi, pj = particles[i], particles[j]
            delta = pj.pos - pi.pos
            dist = np.linalg.norm(delta)

            if dist < pi.radius + pj.radius:
                overlap = pi.radius + pj.radius - dist
                normal = delta / dist if dist != 0 else np.array([1.0, 0.0])

                # относительная скорость в нормальном и касательном направлениях
                rel_vel = pj.vel - pi.vel
                overlap_rate = np.dot(rel_vel, normal)
                tangential_vel = rel_vel - overlap_rate * normal
                rel_vel_tang = np.cross([0, 0, np.linalg.norm(tangential_vel)], [0, 0, 1])[2]

                # Для простоты считаем, что касательное смещение = 0 (можно расширить)
                tangential_disp = 0.0
                effective_radius = (pi.radius * pj.radius) / (pi.radius + pj.radius)

                fn_vec, ft_vec, torque_i, torque_j = contact_model.compute_forces(
                    overlap,
                    overlap_rate,
                    tangential_disp,
                    rel_vel_tang,
                    effective_radius,
                    normal,
                    pi,
                    pj
                )

                pi.apply_force(-fn_vec - ft_vec, torque_i)
                pj.apply_force(fn_vec + ft_vec, torque_j)

    # ---- Частица‑граница ----
    for boundary in boundaries:
        for p in particles:
            coll = boundary.detect_collision(p)
            if coll is None:
                continue

            overlap, contact_point, normal, overlap_rate, tangential_vel = coll

            # относительная касательная скорость (модуль)
            rel_vel_tang = np.cross([0, 0, np.linalg.norm(tangential_vel)], [0, 0, 1])[2]

            # Считаем, что касательное смещение = 0
            tangential_disp = 0.0
            effective_radius = p.radius

            fn_vec, ft_vec, torque_p, _ = contact_model.compute_forces(
                overlap,
                overlap_rate,
                tangential_disp,
                rel_vel_tang,
                effective_radius,
                normal,
                p,
                None
            )

            p.apply_force(-fn_vec - ft_vec, torque_p)

            if isinstance(boundary, WallCircle):
                # реактивный момент, который частицы передали барабану
                boundary.apply_driving_torque(torque_p)