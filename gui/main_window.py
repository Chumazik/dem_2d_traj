import sys
from PyQt5.QtWidgets import (
    QMainWindow, QSplitter, QApplication, QMenuBar, QMenu, QAction,
    QMessageBox, QToolBar, QStatusBar, QWidget, QVBoxLayout
)
from PyQt5.QtCore import Qt
from .input_widget import InputWidget
from .output_widget import OutputWidget
from .simulation_thread import SimulationThread
from dem.simulation import Simulation

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # ---- Основные виджеты ----
        self.input_widget = InputWidget()
        self.output_widget = OutputWidget()

        # ---- Разделитель для размещения ввода и вывода ----
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.input_widget)
        splitter.addWidget(self.output_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

        # ---- Центральный виджет ----
        central = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(splitter)
        central.setLayout(layout)
        self.setCentralWidget(central)

        self.setWindowTitle("2D DEM Simulation")
        self.resize(1200, 800)  # Устанавливаем разумный размер окна

        # ----- Status bar -----
        self.status = QStatusBar()
        self.setStatusBar(self.status)

        # ----- Поток симуляции -----
        self.sim_thread = SimulationThread()
        self.sim_thread.finished.connect(self.on_simulation_finished)

        # ----- Кнопка запуска -----
        self.input_widget.apply_button.clicked.connect(self.start_simulation)

    def start_simulation(self):
        try:
            config = self.input_widget.get_config()
            simulation = Simulation(config)
            self.sim_thread.setSimulation(simulation)
            self.status.showMessage("Запуск симуляции...")
            self.sim_thread.start()
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось запустить симуляцию:\n{e}")

    def on_simulation_finished(self, simulation: Simulation):
        try:
            trajectories = simulation.get_trajectories()
            self.output_widget.update_particles(trajectories)
            self.output_widget.show_results(simulation)
            self.status.showMessage("Симуляция завершена")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось отобразить результаты:\n{e}")
