from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTabWidget, QLabel, QTextEdit
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class OutputWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        self.tabs = QTabWidget()

        self.trajectories_fig = plt.figure(figsize=(10, 6))
        self.ax1 = self.trajectories_fig.add_subplot(111)
        self.trajectories_canvas = FigureCanvas(self.trajectories_fig)
        self.tabs.addTab(self.trajectories_canvas, "Траектории")

        self.torque_fig = plt.figure(figsize=(10, 6))
        self.ax2 = self.torque_fig.add_subplot(111)
        self.torque_canvas = FigureCanvas(self.torque_fig)
        self.tabs.addTab(self.torque_canvas, "Приводной момент")

        self.average_torque_label = QLabel("Средний момент: 0.0 Н·м")
        self.peak_torque_label = QLabel("Пиковый момент: 0.0 Н·м")
        self.power_label = QLabel("Мощность: 0.0 Вт")

        self.layout.addWidget(self.tabs)
        self.layout.addWidget(self.average_torque_label)
        self.layout.addWidget(self.peak_torque_label)
        self.layout.addWidget(self.power_label)

        self.setLayout(self.layout)

    def update_particles(self, particles):
        self.ax1.clear()
        for particle in particles:
            x_coords = [p[0] for p in particle]
            y_coords = [p[1] for p in particle]
            self.ax1.plot(x_coords, y_coords)
        self.ax1.set_title("Траектории частиц")
        self.ax1.set_xlabel("Позиция X (м)")
        self.ax1.set_ylabel("Позиция Y (м)")
        self.ax1.grid(True)
        self.trajectories_canvas.draw()

    def show_results(self, simulation):
        if hasattr(simulation, 'torque_history') and len(simulation.torque_history) > 0:
            average_torque = sum(simulation.torque_history) / len(simulation.torque_history)
            peak_torque = max(simulation.torque_history)
            power = peak_torque * simulation.config.drum_omega

            self.average_torque_label.setText(f"Средний момент: {average_torque:.2f} Н·м")
            self.peak_torque_label.setText(f"Пиковый момент: {peak_torque:.2f} Н·м")
            self.power_label.setText(f"Мощность: {power:.2f} Вт")
