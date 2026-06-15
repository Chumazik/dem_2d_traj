import matplotlib.pyplot as plt



def plot_trajectories(ax, particles, boundaries):

    for particle in particles:

        ax.plot([p[0] for p in particle.history], [p[1] for p in particle.history])

    for boundary in boundaries:

        if isinstance(boundary, WallCircle):

            circle = plt.Circle(boundary.center, boundary.radius, color='gray', fill=False)

            ax.add_artist(circle)



def plot_torque(ax, time_array, torque_array):

    ax.plot(time_array, torque_array)