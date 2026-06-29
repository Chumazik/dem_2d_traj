import matplotlib.pyplot as plt
import numpy as np
from dem.geometry import WallCircle, Lifter


def plot_trajectories(ax, particles_histories, boundaries):                                                                                                                      
    """Рисует траектории частиц и границы."""                                                                                                                                    
    for traj in particles_histories:                                                                                                                                             
        xs = [p[0] for p in traj]                                                                                                                                                
        ys = [p[1] for p in traj]                                                                                                                                                
        ax.plot(xs, ys, linewidth=0.8)

    for b in boundaries:                                                                                                                                                         
        if isinstance(b, WallCircle):                                                                                                                                            
            circle = plt.Circle(b.center, b.radius, color='gray', fill=False, linewidth=1.5)                                                                                     
            ax.add_artist(circle)
        elif isinstance(b, Lifter):
            pts = np.array([b.p1, b.p2, b.p3, b.p4, b.p1])
            ax.plot(pts[:, 0], pts[:, 1], color='gray', linewidth=1.5)
            
    ax.set_aspect('equal')                                                                                                                                                       
    ax.grid(True) 

def plot_torque(ax, time_array, torque_array):                                                                                                                                   
    ax.plot(time_array, torque_array, color='r')                                                                                                                                 
    ax.set_xlabel('Время, с')                                                                                                                                                    
    ax.set_ylabel('Момент, Н·м')                                                                                                                                                 
    ax.grid(True)
