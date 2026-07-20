from flask import Flask, render_template, request, jsonify
import threading
import time
from dem.simulation import Simulation
from utils.config import SimulationConfig
from web.live_buffer import LiveBuffer

app = Flask(__name__)

# Простое хранилище состояния последней симуляции (для примера)
sim_state = {
    "running": False,
    "progress": 0.0,
    "trajectories": None,        # финальные траектории
    "torque_history": None,
    "time": None,
    "config": None,
    "sim_id": 0
}
sim_lock = threading.Lock()

# Потокобезопасный буфер промежуточных результатов
live_buffer = LiveBuffer()


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
    global sim_state
    with sim_lock:
        if sim_state["running"] and sim_state["sim_id"] != sim_id:
            return
        sim_state["running"] = True
        sim_state["progress"] = 0.0
        sim_state["sim_id"] = sim_id
        sim_state["trajectories"] = None
        sim_state["torque_history"] = None
        sim_state["time"] = None

    sim = Simulation(config)

    # Инициализируем буфер промежуточных результатов
    live_buffer.reset(int(config.num_particles))
    live_buffer.mark_running(True)

    total_steps = int(config.total_time / config.dt)
    step_count = 0
    t = 0.0

    while t < config.total_time and not sim.stop_requested:
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

        with sim_lock:
            if not sim_state["running"] or sim_state["sim_id"] != sim_id:
                break
            sim_state["progress"] = progress

        live_buffer.append(particles, t, torque_now)
        live_buffer.set_last_step(step_count)
        live_buffer.set_progress(progress)

    live_buffer.mark_running(False)

    raw_traj = sim.get_trajectories()
    traj = normalize_trajectories(raw_traj)
    with sim_lock:
        sim_state["running"] = False
        sim_state["trajectories"] = traj
        sim_state["torque_history"] = sim.torque_history if hasattr(sim, "torque_history") else []
        sim_state["time"] = sim.time if hasattr(sim, "time") else []
        sim_state["config"] = config


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/start", methods=["POST"])
def start():
    data = request.json
    sim_id = data.get("sim_id", 0)
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
        lifter_height=float(data.get("lifter_height", 0.0)),
        lifter_width=float(data.get("lifter_width", 0.02)),
        num_lifters=int(data.get("num_lifters", 0)),
        dt=float(data.get("dt", 1e-5)),
        total_time=float(data.get("total_time", 5.0))
    )
    thread = threading.Thread(target=run_simulation, args=(config, sim_id))
    thread.start()
    return jsonify({"status": "started", "sim_id": sim_id})


@app.route("/stop", methods=["POST"])
def stop():
    with sim_lock:
        if sim_state["running"]:
            sim_state["running"] = False
            live_buffer.mark_running(False)
            return jsonify({"status": "stopped"})
        return jsonify({"status": "not_running"})


@app.route("/status")
def status():
    with sim_lock:
        return jsonify({
            "running": sim_state["running"],
            "progress": sim_state["progress"],
            "has_results": sim_state["trajectories"] is not None,
            "sim_id": sim_state["sim_id"]
        })


@app.route("/partial_results")
def partial_results():
    """Промежуточные результаты: уже накопленные на данный момент срезы."""
    sim_id = request.args.get("sim_id", type=int)
    with sim_lock:
        if sim_state["sim_id"] == 0 or sim_id is None or sim_id != sim_state["sim_id"]:
            return jsonify({"error": "no active simulation for this sim_id"})
    snap = live_buffer.snapshot()
    if not snap["trajectories"] and not snap["time"]:
        return jsonify({"error": "no partial data yet"})
    return jsonify(snap)


@app.route("/results")
def results():
    sim_id = request.args.get("sim_id", type=int)
    with sim_lock:
        if sim_id is not None and sim_id != sim_state["sim_id"] and sim_state["trajectories"] is None:
            return jsonify({"error": "no results for this sim_id"})
        if sim_state["trajectories"] is None:
            return jsonify({"error": "no results"})
        return jsonify({
            "trajectories": sim_state["trajectories"],
            "torque_history": sim_state["torque_history"],
            "time": sim_state["time"],
            "config": sim_state["config"].__dict__ if sim_state["config"] else {}
        })
