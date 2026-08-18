# -*- coding: utf-8 -*-
import numpy as np
from dem.simulation import Simulation
from utils.config import SimulationConfig

cfg = SimulationConfig(num_particles=1, total_time=0.01, dt=1e-4,
                       drum_radius=0.0, use_jit=True, gravity=9.81)
sim = Simulation(cfg)
sim.boundaries = []
p = sim.particles[0]
p.pos[:] = 0.0
p.vel[:] = 0.0
p.force[:] = 0.0
y0 = float(p.pos[1])
print(f"Initial: pos={p.pos}, y0={y0}")

for k in range(100):
    sim.step()

t = 100 * 1e-4
y_analytic = 0.5 * 9.81 * (100 * 1e-4)**2
v_analytic = 9.81 * 100 * 1e-4

print(f"After 100 steps (t={100*1e-4:.4f}s):")
print(f"  p.pos = {sim.particles[0].pos}")
print(f"  y0 = {y0:.10f}")
print(f"  y = {sim.particles[0].pos[1]:.10f}")
print(f"  dy = {sim.particles[0].pos[1] - y0:.10f}")
print(f"  expected y = {0.5*9.81*0.01**2:.10f}")
print(f"  diff = {sim.particles[0].pos[1] - y0 - 0.5*9.81*0.01**2:.2e}")
print(f"  v = {sim.particles[0].vel[1]} (analytic = {9.81*100*1e-4})")
print(f"  force = {sim.particles[0].force}")