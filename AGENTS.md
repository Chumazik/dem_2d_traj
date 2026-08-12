# AGENTS.md

2D DEM (discrete element method) simulation of disks in a rotating drum. Physics core in Python + Numba; front end is a Flask web app. Docs/comments are in Russian.

## Commands
- Run app: `python main.py` → Flask dev server on http://localhost:5000
- Tests: `python -m unittest discover -s tests -v` (57 tests, all pass)
- Profile the hot loop: `python -m bench.profile_sim` (slow: 200 particles × 5000 steps on the pure-Python O(N²) path)
- Install reading extras once: `pip install xlrd` (for `instruct/Media Charge_Trajectories.xls`)
- Install GPU extras (optional): `pip install cupy-cuda12x` (matching your CUDA toolkit) — without it, the UI's "GPU" mode silently falls back to Numba.
- No linter / formatter / typecheck config exists; there is no CI (see Git).

## Gotchas
- `requirements.txt` lists `flask` and `numba`, which are required at runtime. Keep it in sync with imports (`main.py`/`web/app.py` use Flask, `dem/jit_kernels/` uses numba).
- `SimulationConfig` (`utils/config.py`) has both `use_jit: bool = True` (Numba JIT, default on) and `use_gpu: bool = False` (CuPy GPU). Compute backend is picked per call by `dem/force_calculation.py` and `dem/integrator.py` in this order: **GPU (CuPy) → Numba JIT → pure Python**. Falbacks are silent `try/except`. If CUDA is missing, `dem.gpu_backend.is_available()` returns `False` and the GPU branch is skipped.
- `SimulationConfig.gravity: float = 9.81` (m/s²) is applied as a body force in `compute_all_forces`. Direction is along `+Y` in the simulation/canvas frame (which is "downward" on the page). `compute_all_forces` resets `particle.force` at the top before adding anything, so the gravity contribution does not stack on leftover integrator force from a previous step.
- Начальная упаковка частиц (``Simulation.initialize_particles``) делает **осевшую кучу на дне** барабана с нулевой начальной скоростью, согласованно со статическим превью. Параметры:
    * `gap_fraction: 0.05` — доля дополнительного межцентрового зазора (`spacing = d·(1+gap)`);
    * `angle_of_repose_deg: 35.0` — угол естественного откоса (определяет наклон верха кучи);
    * `apparent_mill_filling: 28.0` — % (определяет ширину основания кучи).
  Python-упаковка и JS-превью используют один и тот же алгоритм (settled hex heap), поэтому старт симуляции визуально совпадает с превью.

## Модельные инварианты (теоретическая проверка)

Согласно дем-теории (Cundall & Strack 1979, MercuryDPM ref-manual) и
тесту ``tests/test_forces_inventory.py``:

- **Сцена строго 2D.** Ни в одном из ``dem/*.py`` нет обращений к 3-й
  оси координат (``pos[:, 2]``, ``shape=(3, ...)`` и т.п.). Все позиции,
  скорости и силы — двумерные массивы длины 2.
- **Единственная внешняя сила — гравитация** ``+Y``, инвариантно
  фиксируемая в ``compute_all_forces``: ``p.force[1] += p.mass · g``.
  Других ``p.force[…] += `` операторов в ``dem/`` нет, силы приходят только
  через ``Particle.apply_force`` из контактной логики (пар + стенок).
- **Соглашение осей в коде:** ``+X`` — вправо, ``+Y`` — "вниз" в канвасе
  (Y-down web-конвенция); ``g > 0`` толкает частицы в ``+Y``. Это
  физически эквивалентно стандартному Y-up соглашению с ``g`` вдоль
  ``−Y``. Тест ``test_signed_gravity_yields_correct_free_fall`` сравнивает
  симуляцию с аналитическим решением ``y(t) = 0.5·g·t²``, ``v(t) = g·t``
  (на пути Numba и CPU).
- **Контактные силы подчиняются Ньютону 3-му закону.**
  Парный контакт: ``F₁ + F₂ = 0``. Контакт «частица–стенка» сообщает
  частице только нормальную силу, направленную ВНУТРЬ барабана; никакого
  внешнего "толчка" вдоль оси вращения нет ({field}[ ``WallCircle`` /
  ``Lifter``] не имеет метода вроде ``apply_external_force``).
- **Контакт «частица–барабан»** в ``WallCircle.detect_collision`` срабатывает
  только когда поверхность частицы достигает ВНУТРЕННЕЙ стенки:
  ``R - r < dist < R + r``, где ``overlap = dist + r - R``. Пока центр
  частицы находится строго внутри (``dist <= R - r``) — контакта нет.
  Раньше здесь использовался заведомо неверный ``overlap = R + r - dist``
  с условием ``dist <= R + r``, из-за чего «сидящая» на стенке частица
  получала ложное перекрытие на полный радиус и нефизичный энерговброс
  (разлет частиц, скорости ~200 м/с, силы ~200 кН). Исправлено.
- **Тангенциальное (статическое/кинетическое) трение** в парных контактах
  накапливает касательное смещение ``td += v_t·dt`` вдоль касательного
  направления; касательная сила = ``−k_t·td − γ_t·v_t`` с кулоновским
  пределом ``μ·|F_n|``. Исправлено и в Numba-ядре
  ``dem/jit_kernels/pairwise.py``, и в Python-путях
  (``_pairwise_python_loop``, ``_compute_pairwise_forces_python``) — JIT и
  Python-пути согласованы (даёт одинаковую силу трения). Раньше в Numba-ядре
  накопления не было (``0.0*dt``) и не было вязкого члена ``−γ_t·v_t``.

Библиография по теории: сопр. документы в ``instruct/``, см. также
[Wikipedia: Discrete element method](https://en.wikipedia.org/wiki/Discrete_element_method)
(MercuryDPM reference manual в архиве https://tiplath.github.io/docs.mercurydpm.github.io/).
- `Simulation.step()` tracks its own `_sim_time` clock (incremented by `config.dt` per call) so lifter rotations advance correctly when `step()` is called directly from external loops (web app `run_simulation`, tests). Without this, lifters stayed at `base_angle` because `self.time` was never appended.
- The DEM particle-particle and particle-wall contact law is embedded in `dem/contact_model.py::ContactModel.compute_forces` and is used both by the CPU loop and the Numba `_pairwise_particle_forces` kernel. Newton's 3rd law is enforced by the kernel (each pair contributes opposite `force_out[i]` / `force_out[j]`).
- Stability caveat: with very stiff contacts (large `kn`) or under large `drum_omega`, the explicit time-stepping can become numerically unstable, especially with the pure-Python integrator. If particles "escape" the drum in the live view, lower `dt` and/or `kn`.
- Nested `dem_project/` is a stale copy with its own `.git` — do not edit; the outer repo tracks it as untracked.

## Architecture
- `dem/` physics: `simulation.py` (`Simulation.step()` is the hot loop, with internal `_sim_time` clock for lifter rotation), `particle.py`, `contact_model.py`, `geometry.py` (rotating drum + lifters), `force_calculation.py`, `integrator.py`, `jit_kernels/` (Numba kernels: `pairwise`, `wall`, `integrator`), `analytical.py` (single-ball trajectory analytical model — see Analytical tab), `gpu_backend.py` (CuPy kernel stubs + `is_available()`).
- Simulation hot loop uses `compute_all_forces` (pairwise → batched GPU/Numba/CPU; walls → CPU; gravity added last) + `velocity_verlet_step` from `dem/integrator.py` (also GPU→Numba→CPU chain). `compute_all_forces` reads `contact_model.config.use_jit`/`use_gpu` to select the pairwise backend and the gravity.
- Numba JIT and pure-Python integrator paths are kept in sync (both do a full 2-step Velocity Verlet + `update_history()`); verify parity after changing either (`dem/integrator.py`).
- Analytical module (`dem/analytical.py`) reproduces the inputs/outputs of `instruct/Media Charge_Trajectories.xls` (Moly-Cop Tools): critical/operating speed, shoulder breakaway, parabolic flight, impact point/velocity/energy, filling-based kidney/toe/clock, torque + power. Classical breakaway (`acos φ²`) and shoulder angles from the spreadsheet differ because the spreadsheet's lift-off model includes rolling + sliding on the lifter (Moly-Cop proprietary, computed via Solver). Numerical match should hold for `critical_speed_rpm`, `impact_speed_ft_s`, `impact_kinetic_energy_joules`; see `diff_vs_molycop` field in the response for a per-metric comparison.
- `web/` Flask app: `app.py` runs the simulation in a background thread; `SimState` + `LiveBuffer` share a single `threading.RLock`; endpoints: `/` (page with two tabs), `/start` (accepts `use_jit`, `use_gpu`, `gravity`), `/stop`, `/status`, `/partial_results` (accepts optional `tail=N` to return only the last N frames — used by the live canvas), `/results`, **`/analytical` (POST)** — accepts JSON with the Moly-Cop input set and returns the full `dem.analytical.AnalyticalOutputs` (~60 fields + a 41-point trajectory).
- UI has two tabs: "DEM симуляция" (existing) and "Аналитика траекторий" (form + Plotly trajectory plot + results/diff tables).
- The DEM tab's `#preview` canvas is a geometry preview before launching, and is **automatically replaced by a live simulation viewer** while a run is in progress — it polls `/partial_results?tail=1` every 250 ms and renders drum circle, rotated lifters (`base_angle + drum_omega * timeNow`), and the latest particle positions with a step/progress/time/max-contact-force/max-speed HUD overlay. Plotly plots (`#traj-plot`, `#torque-plot`, `#dynamics-plot`) still show the full trajectory/moment/dynamics history after the run finishes.
- Lifters are drawn **inside the drum** (anchored on the drum surface, extending inward by `lifter_height`). Static-preview particle packing illustrates a settled-at-bottom heap whose width respects the apparent mill filling % and whose slope respects `angle_of_repose_deg`.
- The DEM tab has a "Вычислитель" select with three options: `cpu_jit` (default, Numba JIT), `cpu` (pure Python), `gpu` (CuPy with auto-fallback to Numba). The selected mode is sent to `/start` as `use_jit`+`use_gpu`; the response echo + `gpu_available` flag is shown next to the dropdown.
- `gui/` is the legacy PyQt5 GUI (unused by `main.py`); `bench/` profiling; `utils/config.py` defines `SimulationConfig`. `instruct/recom` has pending optimization notes for `web/live_buffer.py`.

## Git
- Single branch `main`; remote: `https://github.com/Chumazik/dem_2d_traj.git`.
- `.github/`, `*.json`, `mcp-server/`, `.continue/` are gitignored. `.github/workflows/` are Qwen AI triage workflows, not CI that runs tests.
