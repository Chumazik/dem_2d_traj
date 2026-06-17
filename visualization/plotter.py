import matplotlib.pyplot as plt
from .visualization import WallCircle


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
            
    ax.set_aspect('equal')                                                                                                                                                       
    ax.grid(True) 

def plot_torque(ax, time_array, torque_array):                                                                                                                                   
                                                                                                                                                                                                                                                                         
    ax.plot(time_array, torque_array, color='r')                                                                                                                                 
    ax.set_xlabel('Время, с')                                                                                                                                                    
    ax.set_ylabel('Момент, Н·м')                                                                                                                                                 
    ax.grid(True)         