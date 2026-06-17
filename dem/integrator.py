import numpy as np
from .force_calculation import compute_all_forces

def velocity_verlet_step(particles, dt, contact_model, boundaries):
    """
    Один шаг интегратора Velocity Verlet.
    """
    # ----- Полушаг скоростей -----
    for p in particles:
        a = p.force / p.mass
        alpha = p.torque / p.inertia
        p.vel += 0.5 * a * dt
        p.ang_vel += 0.5 * alpha * dt

    # ----- Обновление позиций -----
    for p in particles:
        p.pos += p.vel * dt
        # угловая позиция в 2D не хранится, но учитываем вращение в расчётах
        p.ang_vel += p.ang_vel * dt  # (можно опустить, оставлено для совместимости)

    # ----- Обнуление сил -----
    for p in particles:
        p.reset_force()

    # ----- Вычисление новых сил -----
    compute_all_forces(particles, boundaries, contact_model)

    # ----- Второй полушаг скоростей -----
    for p in particles:
        a = p.force / p.mass
        alpha = p.torque / p.inertia
        p.vel += 0.5 * a * dt
        p.ang_vel += 0.5 * alpha * dt

    # ----- Сохранение истории -----
    for p in particles:
        p.update_history()