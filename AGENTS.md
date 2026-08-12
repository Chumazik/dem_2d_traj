# AGENTS.md

2D DEM (discrete element method) simulation of disks in a rotating drum. Physics core in Python + Numba; front end is a Flask web app. Docs/comments are in Russian.

## Commands
- Run app: `python main.py` → Flask dev server on http://localhost:5000
- Tests: `python -m unittest discover -s tests -v` (33 tests, all pass)
- Profile the hot loop: `python -m bench.profile_sim` (slow: 200 particles × 5000 steps on the pure-Python O(N²) path)
- Install reading extras once: `pip install xlrd` (for `instruct/Media Charge_Trajectories.xls`)
- Install GPU extras (optional): `pip install cupy-cuda12x` (matching your CUDA toolkit) — without it, the UI's "GPU" mode silently falls back to Numba.
- No linter / formatter / typecheck config exists; there is no CI (see Git).

## Gotchas
- `requirements.txt` lists `flask` and `numba`, which are required at runtime. Keep it in sync with imports (`main.py`/`web/app.py` use Flask, `dem/jit_kernels/` uses numba).
- `SimulationConfig` (`utils/config.py`) has both `use_jit: bool = True` (Numba JIT, default on) and `use_gpu: bool = False` (CuPy GPU). Compute backend is picked per call by `dem/force_calculation.py` and `dem/integrator.py` in this order: **GPU (CuPy) → Numba JIT → pure Python**. Falbacks are silent `try/except`. If CUDA is missing, `dem.gpu_backend.is_available()` returns `False` and the GPU branch is skipped.
- Nested `dem_project/` is a stale copy with its own `.git` — do not edit; the outer repo tracks it as untracked.

## Architecture
- `dem/` physics: `simulation.py` (`Simulation.step()` is the hot loop), `particle.py`, `contact_model.py`, `geometry.py` (rotating drum + lifters), `force_calculation.py`, `integrator.py`, `jit_kernels/` (Numba kernels: `pairwise`, `wall`, `integrator`), `analytical.py` (single-ball trajectory analytical model — see Analytical tab), `gpu_backend.py` (CuPy kernel stubs + `is_available()`).
- Simulation hot loop uses `compute_all_forces` (pairwise → batched GPU/Numba/CPU; walls → CPU) + `velocity_verlet_step` from `dem/integrator.py` (also GPU→Numba→CPU chain). `compute_all_forces` reads `contact_model.config.use_jit`/`use_gpu` to select the pairwise backend.
- Numba JIT and pure-Python integrator paths are kept in sync (both do a full 2-step Velocity Verlet + `update_history()`); verify parity after changing either (`dem/integrator.py`).
- Analytical module (`dem/analytical.py`) reproduces the inputs/outputs of `instruct/Media Charge_Trajectories.xls` (Moly-Cop Tools): critical/operating speed, shoulder breakaway, parabolic flight, impact point/velocity/energy, filling-based kidney/toe/clock, torque + power. Classical breakaway (`acos φ²`) and shoulder angles from the spreadsheet differ because the spreadsheet's lift-off model includes rolling + sliding on the lifter (Moly-Cop proprietary, computed via Solver). Numerical match should hold for `critical_speed_rpm`, `impact_speed_ft_s`, `impact_kinetic_energy_joules`; see `diff_vs_molycop` field in the response for a per-metric comparison.
- `web/` Flask app: `app.py` runs the simulation in a background thread; `SimState` + `LiveBuffer` share a single `threading.RLock`; endpoints: `/` (page with two tabs), `/start` (accepts `use_jit` and `use_gpu`), `/stop`, `/status`, `/partial_results` (accepts optional `tail=N` to return only the last N frames — used by the live canvas), `/results`, **`/analytical` (POST)** — accepts JSON with the Moly-Cop input set and returns the full `dem.analytical.AnalyticalOutputs` (~60 fields + a 41-point trajectory).
- UI has two tabs: "DEM симуляция" (existing) and "Аналитика траекторий" (form + Plotly trajectory plot + results/diff tables).
- The DEM tab's `#preview` canvas is a geometry preview before launching, and is **automatically replaced by a live simulation viewer** while a run is in progress — it polls `/partial_results?tail=1` every 250 ms and renders drum circle, rotated lifters (`base_angle + drum_omega * timeNow`), and the latest particle positions with a step/progress/time HUD overlay. Plotly plots (`#traj-plot`, `#torque-plot`) still show the full trajectory/moment history after the run finishes.
- The DEM tab has a "Вычислитель" select with three options: `cpu_jit` (default, Numba JIT), `cpu` (pure Python), `gpu` (CuPy with auto-fallback to Numba). The selected mode is sent to `/start` as `use_jit`+`use_gpu`; the response echo + `gpu_available` flag is shown next to the dropdown.
- `gui/` is the legacy PyQt5 GUI (unused by `main.py`); `bench/` profiling; `utils/config.py` defines `SimulationConfig`. `instruct/recom` has pending optimization notes for `web/live_buffer.py`.

## Git
- Single branch `main`; remote: `https://github.com/Chumazik/dem_2d_traj.git`.
- `.github/`, `*.json`, `mcp-server/`, `.continue/` are gitignored. `.github/workflows/` are Qwen AI triage workflows, not CI that runs tests.
