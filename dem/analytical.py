"""Аналитический расчёт траекторий мелющих тел во вращающемся барабане.

Основа модели — классическая теория траекторий одиночного шара (см.
Moly-Cop Tools: Media Charge Trajectories, Alvaro Videla L.; Morrell
"Power Draw of Wet Tumbling Mills"), расширенная выводом максимального
набора выходных данных:

* Критическая скорость и рабочая скорость барабана
* Углы отрыва и падения, радиусы траектории
* Полная таблица полёта (точка–время) для построения графика
* Скорость и кинетическая энергия в точке удара
* Оценка мощности, потребляемой зарядом
* Производные от заполнения (kidney angle, toe angle, clock-эквиваленты)

Все формулы выражены в футах/дюймах/градусах для соответствия входным
данным электронной таблицы ``instruct/Media Charge_Trajectories.xls``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import List, Tuple


G_FT_S2 = 32.17405         # ускорение свободного падения, ft/s²
LB_PER_IN3_STEEL = 0.284   # плотность стали, lb/in³ (типичная для шаров)
J_PER_FTLBF = 1.3558179483314004
KG_PER_LB = 0.45359237
M_PER_FT = 0.3048
IN_PER_FT = 12.0


# ---------------------------------------------------------------------------
# Входные параметры
# ---------------------------------------------------------------------------
@dataclass
class AnalyticalParams:
    """Входные параметры аналитического расчёта (как в таблице Moly-Cop)."""

    effective_mill_diameter_ft: float = 36.0    # эффективный диаметр, ft
    ball_diameter_in: float = 5.0               # диаметр шара, in
    static_friction: float = 0.05              # μs
    dynamic_friction: float = 0.20              # μd
    lifter_face_angle_deg: float = 15.0         # угол передней грани лифтера
    lifter_height_in: float = 8.0               # высота лифтера, in
    pct_critical_speed: float = 76.0            # % от критической
    apparent_mill_filling: float = 28.0         # заполнение, %
    angle_of_repose_deg: float = 35.0           # угол естественного откоса
    # Дополнительные опциональные параметры
    ball_density_lb_in3: float = LB_PER_IN3_STEEL   # материал шара
    g_ft_s2: float = G_FT_S2


# ---------------------------------------------------------------------------
# Выходные данные
# ---------------------------------------------------------------------------
@dataclass
class AnalyticalOutputs:
    """Все производные от входных параметров (максимум выходных данных)."""

    # Геометрия мельницы
    mill_diameter_ft: float = 0.0
    mill_radius_ft: float = 0.0
    mill_circumference_ft: float = 0.0
    mill_cross_section_ft2: float = 0.0
    ball_diameter_ft: float = 0.0
    ball_volume_in3: float = 0.0
    ball_mass_lbm: float = 0.0
    ball_mass_slug: float = 0.0
    ball_mass_kg: float = 0.0
    ball_cross_section_ft2: float = 0.0  # сечение шара в долях площади сечения барабана
    effective_ball_radius_ft: float = 0.0  # r = центр шара на окружности барабана
    lifter_height_ft: float = 0.0
    lifter_face_angle_rad: float = 0.0

    # Скорости
    critical_speed_rad_s: float = 0.0
    critical_speed_rpm: float = 0.0
    operating_speed_rad_s: float = 0.0
    operating_speed_rpm: float = 0.0
    fraction_of_critical: float = 0.0
    mill_peripheral_speed_ft_s: float = 0.0
    ball_peripheral_speed_ft_s: float = 0.0   # при r
    centrifugal_at_r_g: float = 0.0   # ω²·r/g в долях g

    # Углы и позиции (от вертикали вверх, по направлению вращения)
    shoulder_angle_rad: float = 0.0   # acos(φ²) — классический угол отрыва
    shoulder_angle_deg: float = 0.0
    shoulder_position_x_ft: float = 0.0
    shoulder_position_y_ft: float = 0.0
    shoulder_clock_hours: float = 0.0

    apex_time_s: float = 0.0
    apex_height_ft: float = 0.0
    apex_position_x_ft: float = 0.0

    launch_velocity_x_ft_s: float = 0.0
    launch_velocity_y_ft_s: float = 0.0
    launch_speed_ft_s: float = 0.0

    flight_time_s: float = 0.0
    impact_position_x_ft: float = 0.0
    impact_position_y_ft: float = 0.0
    impact_distance_from_center_ft: float = 0.0
    impact_velocity_x_ft_s: float = 0.0
    impact_velocity_y_ft_s: float = 0.0
    impact_speed_ft_s: float = 0.0
    impact_angle_from_vertical_deg: float = 0.0   # для точки удара от вертикали вниз
    impact_angle_from_tangent_deg: float = 0.0    # угол вектора скорости относительно касательной
    impact_kinetic_energy_ftlbf: float = 0.0
    impact_kinetic_energy_joules: float = 0.0

    # Заполнение (kidney/toe-углы и clock-эквиваленты)
    filling_fraction: float = 0.0
    repose_angle_rad: float = 0.0
    kidney_angle_rad: float = 0.0
    kidney_angle_deg: float = 0.0
    toe_angle_rad: float = 0.0
    toe_angle_deg: float = 0.0
    toe_clock_hours: float = 0.0
    charge_void_fraction: float = 0.0     # оценочная пористость заряда
    charge_volume_ft3: float = 0.0        # объём заряда (без пустот)
    charge_mass_lbm: float = 0.0
    charge_mass_kg: float = 0.0
    charge_kinetic_energy_joules: float = 0.0   # суммарная кинетическая энергия заряда в полёте

    # Мощность (Torque ≈ M·g·R·sin(θ_discharge) — упрощённая модель)
    discharge_angle_rad: float = 0.0
    discharge_angle_deg: float = 0.0
    torque_due_to_charge_ftlbf: float = 0.0
    torque_due_to_charge_nm: float = 0.0
    power_draw_hp: float = 0.0
    power_draw_kw: float = 0.0

    # Траектория (таблица для графика)
    trajectory: List[Tuple[float, float, float, float]] = field(default_factory=list)
    # время (с), x (ft), y (ft), V (ft/s)

    # Сравнение со значениями Moly-Cop (диагностика модели)
    diff_vs_molycop: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Вычисления
# ---------------------------------------------------------------------------
def _kg_to_lbm(kg: float) -> float:
    return kg / KG_PER_LB


def _lbm_to_slug(lbm: float, g: float = G_FT_S2) -> float:
    return lbm / g


def compute_analytical(
    params: AnalyticalParams, n_traj_points: int = 21
) -> AnalyticalOutputs:
    """Рассчитать полный набор аналитических показателей и таблицу траектории."""

    out = AnalyticalOutputs()

    # --- Преобразование единиц ---
    D = params.effective_mill_diameter_ft
    d_in = params.ball_diameter_in
    d_ft = d_in / IN_PER_FT
    r = D / 2.0 - d_ft / 2.0  # радиус центра шара при контакте

    out.mill_diameter_ft = D
    out.mill_radius_ft = D / 2.0
    out.mill_circumference_ft = math.pi * D
    out.mill_cross_section_ft2 = math.pi * (D / 2.0) ** 2
    out.ball_diameter_ft = d_ft
    out.ball_volume_in3 = (4.0 / 3.0) * math.pi * (d_in / 2.0) ** 3
    out.ball_mass_lbm = params.ball_density_lb_in3 * out.ball_volume_in3
    out.ball_mass_kg = out.ball_mass_lbm * KG_PER_LB
    out.ball_mass_slug = _lbm_to_slug(out.ball_mass_lbm, params.g_ft_s2)
    out.ball_cross_section_ft2 = math.pi * (d_ft / 2.0) ** 2
    out.effective_ball_radius_ft = r
    out.lifter_height_ft = params.lifter_height_in / IN_PER_FT
    out.lifter_face_angle_rad = math.radians(params.lifter_face_angle_deg)

    g = params.g_ft_s2

    # --- Критическая скорость:  ω_c = √(g / r) ---
    w_c = math.sqrt(g / r)
    out.critical_speed_rad_s = w_c
    out.critical_speed_rpm = w_c * 60.0 / (2.0 * math.pi)

    # --- Рабочая скорость ---
    phi = params.pct_critical_speed / 100.0
    out.fraction_of_critical = phi
    w = phi * w_c
    out.operating_speed_rad_s = w
    out.operating_speed_rpm = w * 60.0 / (2.0 * math.pi)
    out.mill_peripheral_speed_ft_s = w * (D / 2.0)
    out.ball_peripheral_speed_ft_s = w * r
    out.centrifugal_at_r_g = (w * w * r) / g   # = φ² безразмерно

    # --- Классический угол отрыва (shoulder) ---
    cos_theta_s = max(min(phi * phi, 1.0), -1.0)
    theta_s = math.acos(cos_theta_s)
    out.shoulder_angle_rad = theta_s
    out.shoulder_angle_deg = math.degrees(theta_s)
    sx = r * math.sin(theta_s)
    sy = r * math.cos(theta_s)
    out.shoulder_position_x_ft = sx
    out.shoulder_position_y_ft = sy
    out.shoulder_clock_hours = (theta_s / (2.0 * math.pi)) * 12.0

    # --- Стартовая скорость (касательная, против часовой стрелки) ---
    v_t = w * r
    vxs = -v_t * math.cos(theta_s)
    vys = v_t * math.sin(theta_s)
    out.launch_velocity_x_ft_s = vxs
    out.launch_velocity_y_ft_s = vys
    out.launch_speed_ft_s = math.sqrt(vxs * vxs + vys * vys)

    # --- Полёт (парабола, g направлено вниз) ---
    cx, cy = 0.0, 0.0
    R = D / 2.0

    # Параметрическое время: y(t) = sy + vys·t − 0.5·g·t²
    # x(t) = sx + vxs·t
    # x²+y² = R² → квартичное уравнение
    # Численное решение: пересечение с R
    def shell_dist(t: float) -> float:
        x = sx + vxs * t
        y = sy + vys * t - 0.5 * g * t * t
        return x * x + y * y

    R2 = R * R
    A_y = vys
    B_y = sy
    C = -0.5 * g
    # Апекс: vy=0 → t_apex = -A_y/(2C)
    t_apex = -A_y / (2.0 * C)
    x_apex = sx + vxs * t_apex
    y_apex = B_y + A_y * t_apex + C * t_apex * t_apex
    out.apex_time_s = t_apex
    out.apex_height_ft = y_apex
    out.apex_position_x_ft = x_apex

    # Поиск первого пересечения с оболочкой после старта
    t_cross = _first_crossing(R2, shell_dist, t_lo=1e-4, t_hi=5.0)
    if t_cross is None:
        # в качестве запасного варианта — широкий диапазон
        t_cross = _first_crossing(R2, shell_dist, t_lo=1e-4, t_hi=50.0)
    t_imp = t_cross if t_cross is not None else float("nan")
    xi = sx + vxs * t_imp
    yi = B_y + A_y * t_imp + C * t_imp * t_imp
    vxi = vxs
    vyi = A_y + 2.0 * C * t_imp

    out.flight_time_s = t_imp
    out.impact_position_x_ft = xi
    out.impact_position_y_ft = yi
    out.impact_distance_from_center_ft = math.sqrt(xi * xi + yi * yi)
    out.impact_velocity_x_ft_s = vxi
    out.impact_velocity_y_ft_s = vyi
    out.impact_speed_ft_s = math.sqrt(vxi * vxi + vyi * vyi)

    # Угол точки удара относительно вертикали вниз (по направлению вращения)
    if abs(yi) < 1e-9:
        out.impact_angle_from_vertical_deg = 90.0 if xi > 0 else -90.0
    else:
        sign = 1.0 if xi >= 0 else -1.0
        out.impact_angle_from_vertical_deg = sign * math.degrees(
            math.atan2(abs(xi), -yi)
        )

    # Угол вектора скорости относительно касательной к оболочке в точке удара
    if abs(xi) > 1e-9 or abs(yi) > 1e-9:
        nx, ny = xi / out.impact_distance_from_center_ft, yi / out.impact_distance_from_center_ft
        # tangent (CCW)
        tx, ty = -ny, nx
        vnorm = out.impact_speed_ft_s if out.impact_speed_ft_s > 0 else 1.0
        cos_a = (vxi * tx + vyi * ty) / vnorm
        cos_a = max(min(cos_a, 1.0), -1.0)
        out.impact_angle_from_tangent_deg = math.degrees(math.acos(cos_a))

    # Энергия удара на один шар
    KE_ftlbf = 0.5 * out.ball_mass_slug * out.impact_speed_ft_s ** 2
    out.impact_kinetic_energy_ftlbf = KE_ftlbf
    out.impact_kinetic_energy_joules = KE_ftlbf * J_PER_FTLBF

    # --- Заполнение / kidney / toe ---
    out.filling_fraction = params.apparent_mill_filling / 100.0
    out.repose_angle_rad = math.radians(params.angle_of_repose_deg)

    # Заполнение по уравнению площади сегмента:
    # fill = (θ - sin θ)/(2π), где θ — центральный угол от верхней точки до
    # свободной поверхности (kidney angle).
    # Решаем относительно θ итеративно.
    theta_kidney = _solve_filling_angle(out.filling_fraction, repose_angle=out.repose_angle_rad)
    out.kidney_angle_rad = theta_kidney
    out.kidney_angle_deg = math.degrees(theta_kidney)
    # Точка поверхности заряда (на свободной стороне угла kidney):
    surface_x = (D / 2.0 - d_ft / 2.0) * math.sin(theta_kidney)
    surface_y = -(D / 2.0 - d_ft / 2.0) * math.cos(theta_kidney)

    # Toe-угол (классическая формула, Rajamani & Mishra):
    # toe_angle = arccos(φ²) − доп. угол от заполнения
    toe = math.acos(phi * phi) - 0.5 * theta_kidney
    out.toe_angle_rad = toe
    out.toe_angle_deg = math.degrees(toe)
    out.toe_clock_hours = (toe / (2.0 * math.pi)) * 12.0

    out.charge_void_fraction = 0.4   # типичная оценка для шаровой загрузки
    out.charge_volume_ft3 = out.mill_cross_section_ft2 * out.filling_fraction
    charge_mass_lbm = (
        out.charge_volume_ft3 * (12 ** 3)  # ft³ → in³
        * (1.0 - out.charge_void_fraction)
        * params.ball_density_lb_in3
    )
    out.charge_mass_lbm = charge_mass_lbm
    out.charge_mass_kg = charge_mass_lbm * KG_PER_LB
    out.charge_kinetic_energy_joules = (
        KE_ftlbf * (charge_mass_lbm / out.ball_mass_lbm) * J_PER_FTLBF
    )

    # Discharge angle — стандартная модель (≈ угол между центром масс и shoulder)
    out.discharge_angle_rad = theta_kidney / 2.0
    out.discharge_angle_deg = math.degrees(out.discharge_angle_rad)
    # Момент от веса заряда
    out.torque_due_to_charge_ftlbf = charge_mass_lbm * (D / 2.0) * math.sin(
        out.discharge_angle_rad
    )
    out.torque_due_to_charge_nm = out.torque_due_to_charge_ftlbf * 1.355817948
    # Мощность = T · ω (ft·lbf/s → hp: /550)
    power_ftlbf_s = out.torque_due_to_charge_ftlbf * w
    out.power_draw_hp = power_ftlbf_s / 550.0
    out.power_draw_kw = out.power_draw_hp * 0.745699872

    # --- Таблица полёта ---
    trajectory = []
    for k in range(n_traj_points):
        t = (k / (n_traj_points - 1)) * t_imp if (n_traj_points > 1) else 0.0
        x = sx + vxs * t
        y = B_y + A_y * t + C * t * t
        v = math.sqrt(vxs * vxs + (A_y + 2 * C * t) ** 2)
        trajectory.append((t, x, y, v))
    out.trajectory = trajectory

    # --- Диагностика относительно значений из таблицы Moly-Cop ---
    out.diff_vs_molycop = {
        "Nc_rpm": {
            "analytic": out.critical_speed_rpm,
            "sheet": 12.841195418,
        },
        "shoulder_deg": {
            "analytic": out.shoulder_angle_deg,
            "sheet": 42.953433412,
            "note": "Sheet shoulder is post-lifter roll/slide (Moly-Cop proprietary).",
        },
        "impact_toe_deg": {
            "analytic": out.impact_angle_from_vertical_deg,
            "sheet": 38.824223933,
            "note": "Sheet toe is after kidney-angle correction in Moly-Cop.",
        },
        "impact_V_ft_s": {
            "analytic": out.impact_speed_ft_s,
            "sheet": 45.119539,
        },
        "impact_E_J": {
            "analytic": out.impact_kinetic_energy_joules,
            "sheet": 787.064776,
        },
    }
    return out


# ---------------------------------------------------------------------------
# Вспомогательные
# ---------------------------------------------------------------------------
def _solve_filling_angle(fill: float, repose_angle: float, tol: float = 1e-7,
                        max_iter: int = 200) -> float:
    """Решить уравнение (θ − sin θ)/(2π) = fill методом Ньютона/бисекции.

    Вспомогательный аргумент ``repose_angle`` сейчас не используется
    (поверхность заряда считается плоской); оставлен для совместимости.
    """
    del repose_angle  # не используется
    if fill <= 0.0:
        return 0.0
    if fill >= 1.0:
        return 2.0 * math.pi

    def f(theta: float) -> float:
        return (theta - math.sin(theta)) / (2.0 * math.pi) - fill

    lo, hi = 1e-6, 2.0 * math.pi - 1e-6
    flo, fhi = f(lo), f(hi)
    if flo > 0.0:
        return lo
    if fhi < 0.0:
        return hi
    theta = 2.0 * math.pi * fill   # хорошее начальное приближение
    for _ in range(max_iter):
        val = f(theta)
        df = (1.0 - math.cos(theta)) / (2.0 * math.pi)
        if df < 1e-12:
            break
        step = val / df
        theta_new = theta - step
        # ограничим, чтобы не вылететь за границы бисекции
        if theta_new < lo:
            theta_new = (lo + theta) / 2.0
        elif theta_new > hi:
            theta_new = (hi + theta) / 2.0
        if abs(theta_new - theta) < tol * max(1.0, abs(theta)):
            theta = theta_new
            break
        theta = theta_new
    return theta


def _first_crossing(R2: float, dist, t_lo: float, t_hi: float,
                    n_scan: int = 5000) -> float | None:
    """Найти первое пересечение dist(t)=R² после t_lo."""
    pts = [t_lo + (t_hi - t_lo) * i / n_scan for i in range(n_scan + 1)]
    for i in range(len(pts) - 1):
        d0 = dist(pts[i])
        d1 = dist(pts[i + 1])
        if (d0 <= R2 <= d1) or (d0 >= R2 >= d1):
            # линейная интерполяция
            if d1 != d0:
                t = pts[i] + (R2 - d0) / (d1 - d0) * (pts[i + 1] - pts[i])
                return t
            return pts[i]
    return None


def to_jsonable(out: AnalyticalOutputs) -> dict:
    """Преобразовать выходные данные в сериализуемый dict (без кортежей в
    таблице траектории — каждый элемент становится списком)."""
    data = asdict(out)
    data["trajectory"] = [
        {"t": row[0], "x": row[1], "y": row[2], "v": row[3]}
        for row in out.trajectory
    ]
    return data
