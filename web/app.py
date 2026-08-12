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

    def request_stop(self) -> bool:
        """Запрашивает остановку. Возвращает ``True``, если шла симуляция."""
        with self._lock:
            was_running = self._running
            self._running = False
            self._buffer.mark_running(False)
            return was_running

    def start(self, sim_id: int) -> bool:
        """Отмечает старт симуляции. ``False``, если другая уже выполняется."""
        with self._lock:
            if self._running and self._sim_id != sim_id:
                return False
            self._running = True
            self._progress = 0.0
            self._sim_id = sim_id
            self._trajectories = None
            self._torque_history = None
            self._time = None
            return True

    def finalize(self, *, trajectories, torque_history, time, config) -> None:
        """Финализирует симуляцию: записывает результаты и снимает флаг."""
        with self._lock:
            self._running = False
            self._progress = 100.0
            self._trajectories = trajectories
            self._torque_history = torque_history
            self._time = time
            self._config = config
            self._buffer.mark_running(False)
            self._buffer.set_progress(100.0)

    # ----- Снимки под единым локом ----- #
    def status_snapshot(self) -> dict:
        with self._lock:
            return {
                "running": self._running,
                "progress": self._progress,
                "has_results": self._trajectories is not None,
                "sim_id": self._sim_id,
            }

    def results_snapshot(self) -> dict:
        with self._lock:
            if self._trajectories is None:
                return {"error": "no results"}
            return {
                "trajectories": self._trajectories,
                "torque_history": self._torque_history or [],
                "time": self._time or [],
                "config": self._config.__dict__ if self._config else {},
            }

    def partial_snapshot(self, sim_id: int, tail: int = 0) -> dict:
        with self._lock:
            if self._sim_id == 0 or sim_id != self._sim_id:
                return {"error": "no active simulation for this sim_id"}
        snap = self._buffer.snapshot(tail=tail)
        if not snap["trajectories"] and not snap["time"]:
            return {"error": "no partial data yet"}
        return snap

    # ----- Доступ к буферу (для run_simulation) ----- #
    def reset_buffer(self, num_particles: int) -> None:
        with self._lock:
            self._buffer.reset(num_particles)
            self._buffer.mark_running(True)

    def append_point(self, particles, t, torque,
                     step: int, progress: float, running: bool) -> None:
        """Атомарно: добавляет точку и обновляет метаданные."""
        with self._lock:
            self._buffer.append(particles, t, torque,
                                running=running, progress=progress, last_step=step)
            self._progress = progress
            self._running = running

    def should_continue(self, sim_id: int) -> bool:
        with self._lock:
            return self._running and self._sim_id == sim_id

    def stop_requested(self, sim_id: int) -> bool:
        with self._lock:
            return (not self._running) or self._sim_id != sim_id


state = SimState()


def normalize_trajectories(traj):
    """Приводим траектории к списку списков [x, y] по частицам."""
    if traj is None:
        return []
    if isinstance(traj, dict):
        return [list(pos_list) for pos_list in traj.values()]
    if isinstance(traj, list):
        if len(traj) > 0 and hasattr(traj[0], 'history'):
            return [list(p.history) for p in traj]
        return traj
    return []


def run_simulation(config: SimulationConfig, sim_id: int):
    if not state.start(sim_id):
        return

    sim = Simulation(config)
    state.reset_buffer(int(config.num_particles))

    total_steps = max(1, int(config.total_time / config.dt))
    step_count = 0
    t = 0.0

    while t < config.total_time:
        if state.stop_requested(sim_id):
            break
        sim.step()
        step_count += 1
        t += config.dt

        particles = getattr(sim, "particles", None) or []
        torque_now = None
        if hasattr(sim, "torque_history") and sim.torque_history:
            torque_now = sim.torque_history[-1]
        elif hasattr(sim, "torque"):
            torque_now = sim.torque

        progress = min(100.0, (step_count / total_steps) * 100.0)
        # Атомарное обновление буфера и метаданных под единым локом
        state.append_point(particles, t, torque_now,
                           step=step_count, progress=progress,
                           running=state.should_continue(sim_id))

    raw_traj = sim.get_trajectories()
    traj = normalize_trajectories(raw_traj)
    state.finalize(
        trajectories=traj,
        torque_history=sim.torque_history if hasattr(sim, "torque_history") else [],
        time=sim.time if hasattr(sim, "time") else [],
        config=config,
    )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/start", methods=["POST"])
def start():
    data = request.json
    sim_id = int(data.get("sim_id", 0))
    config = SimulationConfig(
        num_particles=int(data.get("num_particles", 100)),
        particle_radius=float(data.get("particle_radius", 0.02)),
        particle_density=float(data.get("particle_density", 2500.0)),
        kn=float(data.get("kn", 1e5)),
        restitution_coeff=float(data.get("restitution", 0.9)),
        friction_static=float(data.get("friction_static", 0.5)),
        friction_dynamic=float(data.get("friction_dynamic", 0.4)),
        rolling_friction=float(data.get("rolling_friction", 0.01)),
        drum_radius=float(data.get("drum_radius", 0.5)),
        drum_omega=float(data.get("drum_omega", 2.0)),
        lifter_height=float(data.get("lifter_height", 0.03)),
        lifter_width=float(data.get("lifter_width", 0.02)),
        num_lifters=int(data.get("num_lifters", 4)),
        dt=float(data.get("dt", 1e-5)),
        total_time=float(data.get("total_time", 5.0)),
        use_jit=_as_bool(data.get("use_jit", True)),
        use_gpu=_as_bool(data.get("use_gpu", False)),
        gravity=float(data.get("gravity", 9.81)),
    )
    thread = threading.Thread(target=run_simulation, args=(config, sim_id))
    thread.start()
    return jsonify({
        "status": "started",
        "sim_id": sim_id,
        "use_jit": config.use_jit,
        "use_gpu": config.use_gpu,
        "gpu_available": _gpu_available(),
        "gravity": config.gravity,
    })


def _as_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(v, (int, float)):
        return bool(v)
    return False


def _gpu_available() -> bool:
    try:
        from dem import gpu_backend
        return gpu_backend.is_available()
    except Exception:
        return False


@app.route("/stop", methods=["POST"])
def stop():
    was_running = state.request_stop()
    return jsonify({"status": "stopped" if was_running else "not_running"})


@app.route("/status")
def status():
    return jsonify(state.status_snapshot())


@app.route("/partial_results")
def partial_results():
    sim_id = request.args.get("sim_id", type=int)
    tail = request.args.get("tail", default=0, type=int)
    return jsonify(state.partial_snapshot(sim_id or 0, tail=max(0, tail)))


@app.route("/results")
def results():
    return jsonify(state.results_snapshot())


@app.route("/analytical", methods=["POST"])
def analytical():
    """Аналитический расчёт траекторий мелющих тел (Moly-Cop-совместимая модель).

    Принимает JSON-тело с теми же параметрами, что и таблица
    ``Media Charge_Trajectories.xls``: эффективный диаметр мельницы,
    диаметр шара, коэффициенты трения, угол и высоту лифтера, % критической
    скорости, заполнение и угол естественного откоса. Возвращает максимум
    производных показателей + координаты траектории для построения графика.
    """
    data = request.get_json(silent=True) or {}
    try:
        params = AnalyticalParams(
            effective_mill_diameter_ft=float(
                data.get("effective_mill_diameter_ft", 36.0)
            ),
            ball_diameter_in=float(data.get("ball_diameter_in", 5.0)),
            static_friction=float(data.get("static_friction", 0.05)),
            dynamic_friction=float(data.get("dynamic_friction", 0.20)),
            lifter_face_angle_deg=float(data.get("lifter_face_angle_deg", 15.0)),
            lifter_height_in=float(data.get("lifter_height_in", 8.0)),
            pct_critical_speed=float(data.get("pct_critical_speed", 76.0)),
            apparent_mill_filling=float(data.get("apparent_mill_filling", 28.0)),
            angle_of_repose_deg=float(data.get("angle_of_repose_deg", 35.0)),
            ball_density_lb_in3=float(data.get("ball_density_lb_in3", 0.284)),
        )
    except (TypeError, ValueError) as exc:
        return jsonify({"error": f"invalid input: {exc}"}), 400

    try:
        n_points = max(2, min(401, int(data.get("n_traj_points", 41))))
        out = compute_analytical(params, n_traj_points=n_points)
    except Exception as exc:
        return jsonify({"error": f"calculation failed: {exc}"}), 500

    return jsonify(to_jsonable(out))
