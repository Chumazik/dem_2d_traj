import numpy as np
from .contact_model import ContactModel, Contact

def compute_all_forces(particles, boundaries, contact_model: ContactModel, contacts=None):
    """
    Вычисляет все взаимодействия частиц‑частиц и частица‑граница.
    Сохраняет реактивный момент в объекте WallCircle.
    """
    n = len(particles)

    if contacts is None:
        contacts = {}

    # ---- Частица-частица ----
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
                tangential_velocity = rel_vel - overlap_rate * normal

                # Создаем или обновляем контакт
                contact_key = (pi.id, pj.id)
                if contact_key not in contacts:
                    contacts[contact_key] = Contact(pi.id, pj.id)

                contact = contacts[contact_key]
                tangential_displacement = contact.tangential_displacement + np.dot(tangential_velocity, normal) * contact_model.config.dt
                effective_radius = (pi.radius * pj.radius) / (pi.radius + pj.radius)

                fn_vec, ft_vec, torque_i, torque_j = contact_model.compute_forces(
                    overlap,
                    overlap_rate,
                    tangential_displacement,
                    np.dot(tangential_velocity, normal),
                    effective_radius,
                    normal,
                    pi,
                    pj
                )

                pi.apply_force(-fn_vec - ft_vec, torque_i)
                pj.apply_force(fn_vec + ft_vec, torque_j)

                # Обновляем касательное смещение
                contact.tangential_displacement = tangential_displacement

    # ---- Частица-граница ----
    for boundary in boundaries:
        for p in particles:
            coll = boundary.detect_collision(p)
            if coll is None:
                continue

            overlap, contact_point, normal, overlap_rate, tangential_velocity = coll

            # относительная касательная скорость (модуль)
            rel_vel_tang = np.dot(tangential_velocity, normal)

            # Создаем или обновляем контакт
            contact_key = (p.id, boundary)
            if contact_key not in contacts:
                contacts[contact_key] = Contact(p.id, None)

            contact = contacts[contact_key]
            tangential_displacement = contact.tangential_displacement + rel_vel_tang * contact_model.config.dt

            fn_vec, ft_vec, torque_p, _ = contact_model.compute_forces(
                overlap,
                overlap_rate,
                tangential_displacement,
                rel_vel_tang,
                p.radius,
                normal,
                p,
                None
            )

            p.apply_force(-fn_vec - ft_vec, torque_p)

            if isinstance(boundary, WallCircle):
                # реактивный момент, который частицы передали барабану
                boundary.apply_driving_torque(torque_p)

    return contacts
