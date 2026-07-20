from flask import Flask, render_template, request, jsonify
import threading
import time
from dem.simulation import Simulation
from utils.config import SimulationConfig

app = Flask(__name__)

# Простое хранилище состояния последней симуляции (для примера)
sim_state = {
    "running": False,
    "progress": 0.0,
    "trajectories": None,
    "torque_history": None,
    "time": None,
    "config": None
}
sim_lock = threading.Lock()

def normalize_trajectories(traj):
    """Приводим траектории к списку списков [x, y] по частицам."""
    if traj is None:
        return []
    # Если это dict с частицами по id: {id: [[x,y],...]}
    if isinstance(traj, dict):
        return [list(pos_list) for pos_list in traj.values()]
    # Если это список частиц с атрибутом history
    if isinstance(traj, list):
        if len(traj) > 0 and hasattr(traj[0], 'history'):
            return [list(p.history) for p in traj]
        return traj
    return []

def run_simulation(config: SimulationConfig):
    global sim_state
    sim = Simulation(config)
    total_steps = int(config.total_time / config.dt)
    step_count = 0
    t = 0.0
    with sim_lock:
        sim_state["running"] = True
        sim_state["progress"] = 0.0
    while t < config.total_time and not sim.stop_requested:
        sim.step()
        step_count += 1
        t += config.dt
        with sim_lock:
            sim_state["progress"] = (step_count / total_steps) * 100.0
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
    thread = threading.Thread(target=run_simulation, args=(config,))
    thread.start()
    return jsonify({"status": "started"})

@app.route("/status")
def status():
    with sim_lock:
        return jsonify({
            "running": sim_state["running"],
            "progress": sim_state["progress"],
            "has_results": sim_state["trajectories"] is not None
        })

@app.route("/results")
def results():
    with sim_lock:
        if sim_state["trajectories"] is None:
            return jsonify({"error": "no results"})
        return jsonify({
            "trajectories": sim_state["trajectories"],
            "torque_history": sim_state["torque_history"],
            "time": sim_state["time"],
            "config": sim_state["config"].__dict__ if sim_state["config"] else {}
        })
