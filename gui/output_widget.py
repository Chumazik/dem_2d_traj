from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTabWidget, QLabel
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

class OutputWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)
        
        # ---- Траектории ----
        self.traj_fig = plt.figure(figsize=(8, 6))
        self.traj_ax = self.traj_fig.add_subplot(111)
        self.traj_canvas = FigureCanvas(self.traj_fig)
        self.tabs.addTab(self.traj_canvas, "Траектории")

        # ---- Приводной момент ----
        self.torque_fig = plt.figure(figsize=(8, 6))
        self.torque_ax = self.torque_fig.add_subplot(111)
        self.torque_canvas = FigureCanvas(self.torque_fig)
        self.tabs.addTab(self.torque_canvas, "Приводной момент")

        # ---- Метки для результатов ----
        self.avg_label = QLabel("Средний момент: Н·м")
        self.peak_label = QLabel("Пиковый момент: Н·м")
        self.power_label = QLabel("Мощность: Вт")

        self.layout.addWidget(self.avg_label)
        self.layout.addWidget(self.peak_label)
        self.layout.addWidget(self.power_label)

    def clear_results(self):
        """Очищает графики и метки."""
        self.traj_ax.clear()
        self.traj_ax.set_title("Траектории частиц")
        self.traj_ax.set_xlabel("X, м")
        self.traj_ax.set_ylabel("Y, м")
        self.traj_ax.grid(True)
        self.traj_canvas.draw()

        self.torque_ax.clear()
        self.torque_ax.set_title("Приводной момент во времени")
        self.torque_ax.set_xlabel("Время, с")
        self.torque_ax.set_ylabel("Момент, Н·м")
        self.torque_ax.grid(True)
        self.torque_canvas.draw()

        self.avg_label.setText("Средний момент: Н·м")
        self.peak_label.setText("Пиковый момент: Н·м")
        self.power_label.setText("Мощность: Вт")

    def update_particles(self, particles_histories):
        """Отрисовывает траектории частиц."""
        self.traj_ax.clear()
        for traj in particles_histories:
            xs = [p[0] for p in traj]
            ys = [p[1] for p in traj]
            self.traj_ax.plot(xs, ys, linewidth=0.8)
        self.traj_ax.set_title("Траектории частиц")
        self.traj_ax.set_xlabel("X, м")
        self.traj_ax.set_ylabel("Y, м")
        self.traj_ax.grid(True)
        self.traj_canvas.draw()

    def show_results(self, simulation):
        if not hasattr(simulation, "torque_history") or len(simulation.torque_history) == 0:
            self.avg_label.setText("Средний момент: Нет данных")
            self.peak_label.setText("Пиковый момент: Нет данных")
            self.power_label.setText("Мощность: Нет данных")
            return

        # Отрисовка графика момента
        self.torque_ax.clear()
        self.torque_ax.plot(simulation.time, simulation.torque_history, color='r')
        self.torque_ax.set_title("Приводной момент во времени")
        self.torque_ax.set_xlabel("Время, с")
        self.torque_ax.set_ylabel("Момент, Н·м")
        self.torque_ax.grid(True)
        self.torque_canvas.draw()

        # Вычисление среднего, пикового момента и мощности
        avg = sum(simulation.torque_history) / len(simulation.torque_history)
        peak = max(simulation.torque_history)
        power = peak * simulation.config.drum_omega

        # Обновление текстов меток
        self.avg_label.setText(f"Средний момент: {avg:.3f} Н·м")
        self.peak_label.setText(f"Пиковый момент: {peak:.3f} Н·м")
        self.power_label.setText(f"Мощность: {power:.3f} Вт")
