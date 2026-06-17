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
            x = [traj[i][0] for traj in self.particles_history]                                                                                                                  
            y = [traj[i][1] for traj in self.particles_history]
            self.scatter = self.ax.scatter(x, y, c='b')
        else:
            x = [traj[i][0] for traj in self.particles_history]                                                                                                                  
            y = [traj[i][1] for traj in self.particles_history]                                                                                                                  
            self.scatter.set_offsets(np.c_[x, y])                                                                                                                                
        return (self.scatter,)

    def create_animation(self, interval=30, repeat=False):                                                                                                                       
        """Возвращает объект FuncAnimation."""                                                                                                                                   
        frames = len(self.particles_history[0])                                                                                                                                  
        ani = animation.FuncAnimation(                                                                                                                                           
            self.fig,                                                                                                                                                            
            self.animate,                                                                                                                                                        
            frames=frames,                                                                                                                                                       
            interval=interval,                                                                                                                                                   
            blit=True,                                                                                                                                                           
            repeat=repeat                                                                                                                                                        
        )
        return ani
