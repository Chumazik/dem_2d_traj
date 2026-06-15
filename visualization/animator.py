import matplotlib.animation as animation
import numpy as np

class Animator:

    def __init__(self, fig, ax, particles_history):
        self.fig = fig
        self.ax = ax
        self.particles_history = particles_history
        self.scatter = None

    def animate(self, i):
        if self.scatter is None:
            self.scatter = self.ax.scatter([p[i][0] for p in self.particles_history], [p[i][1] for p in self.particles_history])
        else:
            x_coords = [p[i][0] for p in self.particles_history]
            y_coords = [p[i][1] for p in self.particles_history]
            self.scatter.set_offsets(np.c_[x_coords, y_coords])
        return self.scatter,

    def create_animation(self):
        ani = animation.FuncAnimation(self.fig, self.animate, frames=len(self.particles_history[0]), blit=True)
        return ani
