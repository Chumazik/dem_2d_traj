# -*- coding: utf-8 -*-
"""Проверка качества физики: куча с лифтерами, долгий прогон, оседание и моменты."""
import numpy as np
from dem.simulation import Simulation
from utils.config import SimulationConfig

cfg = SimulationConfig(num_particles=120, num_lifters=4,
                       lifter_height=0.04, lifter_width=0.02,
                       drum_radius=0.5, particle_radius=0.02,
                       dt=1e-5, total_time=1.0, gravity=9.81,
                       kn=1e5, drum_omega=2.0)
sim = Simulation(cfg)
R = cfg.drum_radius; r = cfg.particle_radius
true_escape = R + r
for k in range(4000):
    sim.step()
maxR = max(np.hypot(p.pos[0], p.pos[1]) for p in sim.particles)
peakV = max(sim.max_velocity_history)
peakF = max(sim.max_force_history)
print(f"4000 steps (t=0.04s): max center R={maxR:.4f} (true_escape={true_escape:.4f})")
print(f"  escaped (center>R+r) = {sum(1 for p in sim.particles if np.hypot(p.pos[0],p.pos[1]) > true_escape)}")
print(f"  peak |v|={peakV:.3f} m/s, peak |F|={peakF:.1f} N")
# Моменты: с лифтерами и вращением должны быть ненулевые
nz = sum(1 for t in sim.torque_history if abs(t) > 1e-9)
print(f"  nonzero torque samples: {nz}/{len(sim.torque_history)}")
print(f"  avg torque (nonzero): {np.mean([t for t in sim.torque_history if abs(t)>1e-9] or [0]):.3f}")