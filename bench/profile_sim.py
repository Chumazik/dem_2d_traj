"""Профилирование DEM-симуляции.

Запускает короткий прогон Simulation и печатает top-25 функций
по суммарному (cumulative) и собственному (tottime) времени.
Используется для оценки узких мест перед оптимизацией (Numba/GPU).

Запуск:
    python -m bench.profile_sim
    # или из корня проекта:
    python bench/profile_sim.py
"""

import cProfile
import io
import pstats
import sys
from pathlib import Path

# Позволяет запускать и из корня, и из bench/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dem.simulation import Simulation
from utils.config import SimulationConfig


def _default_config() -> SimulationConfig:
    """Умеренно тяжёлый прогон: 200 частиц, 0.05 с модельного времени."""
    return SimulationConfig(
        num_particles=200,
        particle_radius=0.02,
        particle_density=2500.0,
        kn=1e5,
        restitution_coeff=0.9,
        friction_static=0.5,
        friction_dynamic=0.4,
        rolling_friction=0.01,
        drum_radius=0.5,
        drum_omega=2.0,
        lifter_height=0.0,
        lifter_width=0.02,
        num_lifters=0,
        dt=1e-5,
        total_time=0.05,
    )


def _format_stats(stats: pstats.Stats, *, sort: str, limit: int = 25) -> str:
    """Возвращает top-N строк в виде текста."""
    buf = io.StringIO()
    stream = pstats.Stats(stats, stream=buf).strip_dirs()
    if sort == "cum":
        stream.sort_stats("cumulative").print_stats(limit)
    elif sort == "tottime":
        stream.sort_stats("tottime").print_stats(limit)
    elif sort == "ncalls":
        stream.sort_stats("ncalls").print_stats(limit)
    else:
        stream.sort_stats("cumulative").print_stats(limit)
    return buf.getvalue()


def run_profile(config: SimulationConfig | None = None,
                sort: str = "cum",
                limit: int = 25) -> str:
    """Прогоняет симуляцию под cProfile и возвращает текстовый отчёт."""
    config = config or _default_config()
    sim = Simulation(config)

    profiler = cProfile.Profile()
    profiler.enable()
    try:
        sim.run()
    finally:
        profiler.disable()

    return _format_stats(profiler, sort=sort, limit=limit)


def main() -> int:
    print(f"Запуск профилирования: "
          f"N={_default_config().num_particles}, "
          f"total_time={_default_config().total_time} c, "
          f"dt={_default_config().dt} c\n")
    report = run_profile()
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
