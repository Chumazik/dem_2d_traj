from flask import Flask, render_template, request, jsonify
import threading
import time
from typing import Optional
from dem.simulation import Simulation
from utils.config import SimulationConfig
from web.live_buffer import LiveBuffer
from dem.analytical import AnalyticalParams, compute_analytical, to_jsonable

app = Flask(__name__)


class SimState:
    """Единое состояние активной симуляции.

    Все обращения к буферу и метаданным (``running``/``progress``/
    ``sim_id``/``trajectories`` и т.д.) синхронизируются одним
    :class:`threading.RLock`, чтобы любой снимок был атомарным.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._running: bool = False
        self._progress: float = 0.0
        self._sim_id: int = 0
        self._trajectories: Optional[list] = None
        self._torque_history: Optional[list] = None
        self._max_force_history: Optional[list] = None
        self._max_velocity_history: Optional[list] = None
        self._time: Optional[list] = None
        self._config: Optional[object] = None
        # Буфер разделяет единый лок с состоянием.
        self._buffer: LiveBuffer = LiveBuffer(lock=self._lock)

    # ----- Метаданные ----- #
    def is_running(self) -> bool:
        with self._lock:
            return self._running

    def progress(self) -> float:
        with self._lock:
            return self._progress

    def sim_id(self) -> int:
        with self._lock:
            return self._sim_id

    def has_results(self) -> bool:
        with self._lock:
            return self._trajectories is not None

    # ----- Управление ----- #
    def start(self, sim_id: int) -> bool:
        with self._lock:
            if self._running:
                return False
            self._running = True
            self._progress = 0.0
            self._sim_id = sim_id
            self._trajectories = None
            self._torque_history = None
            self._max_force_history = None
            self._max_velocity_history = None
            self._time = None
            self._config = None
            return True

    def stop_requested(self, sim_id: int) -> bool:
        with self._lock:
            return self._sim_id != sim_id

    def should_continue(self, sim_id: int) -> bool:
        with self._lock:
            return self._sim_id == sim_id and self._running

    def stop(self) -> None:
        with self._lock:
            self._running = False

    # ----- Буфер ----- #
    def reset_buffer(self, n_particles: int) -> None:
        with self._lock:
            self._buffer.reset(n_particles)

    def append_point(self, particles, t, torque, step, progress, running, max_force, max_velocity):
        with self._lock:
            self._progress = progress
            self._buffer.append(particles, t, torque, step, progress, running, max_force, max_velocity)

    def snapshot(self, tail: Optional[int] = None):
        with self._lock:
            return self._buffer.snapshot(tail)

    # ----- Результаты ----- #
    def finalize(self, trajectories, torque_history, time, max_force_history, max_velocity_history, config):
        with self._lock:
            self._running = False
            self._trajectories = trajectories
            self._torque_history = torque_history
            self._time = time
            self._max_force_history = max_force_history
            self._max_velocity_history = max_velocity_history
            self._config = config

    def get_results(self):
        with self._lock:
            return {
                "trajectories": self._trajectories,
                "torque_history": self._torque_history,
                "time": self._time,
                "max_force_history": self._max_force_history,
                "max_velocity_history": self._max_velocity_history,
                "config": self._config,
            }


state = SimState()


def normalize_trajectories(raw_trajectories):
    """Приводит траектории к формату, пригодному для сериализации в JSON."""
    if not raw_trajectories:
        return []
    return [[[float(x), float(y)] for x, y in traj] for traj in raw_trajectories]


def run_simulation(config: SimulationConfig, sim_id: int):
    if not state.start(sim_id):
        return

    sim = Simulation(config)
    state.reset_buffer(int(config.num_particles))

    step_count = 0
    t = 0.0
    next_output_time = 0.0
    output_dt = config.output_dt if config.adaptive_dt else config.dt

    while t < config.total_time:
        if state.stop_requested(sim_id):
            break
        sim.step()
        step_count += 1
        t = sim._sim_time  # Используем фактическое время симуляции

        particles = getattr(sim, "particles", None) or []
        torque_now = None
        if hasattr(sim, "torque_history") and sim.torque_history:
            torque_now = sim.torque_history[-1]
        elif hasattr(sim, "torque"):
            torque_now = sim.torque

        # Вычисляем текущие max |F| и |v| (см. dem/simulation.py::step).
        max_force_now = (
            float(sim.max_force_history[-1])
            if getattr(sim, "max_force_history", None) else 0.0
        )
        max_velocity_now = (
            float(sim.max_velocity_history[-1])
            if getattr(sim, "max_velocity_history", None) else 0.0
        )

        progress = min(100.0, (t / config.total_time) * 100.0)
        # Аппендим точку в буфер с частотой output_dt для согласованности
        if t >= next_output_time:
            next_output_time = min(t + output_dt, config.total_time)
            state.append_point(particles, t, torque_now,
                             step=step_count, progress=progress,
                             running=state.should_continue(sim_id),
                             max_force=max_force_now,
                             max_velocity=max_velocity_now)

    raw_traj = sim.get_trajectories()
    traj = normalize_trajectories(raw_traj)
    state.finalize(
        trajectories=traj,
        torque_history=sim.torque_history if hasattr(sim, "torque_history") else [],
        time=sim.time if hasattr(sim, "time") else [],
        max_force_history=getattr(sim, "max_force_history", []) or [],
        max_velocity_history=getattr(sim, "max_velocity_history", []) or [],
        config=config,
    )