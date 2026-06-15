import sys
from PyQt5.QtWidgets import QMainWindow, QSplitter, QApplication, QMenuBar, QMenu, QAction, QMessageBox, QToolBar, QStatusBar
from PyQt5.QtCore import Qt
from .simulation_thread import SimulationThread
from .input_widget import InputWidget
from .output_widget import OutputWidget
from dem.simulation import Simulation, SimulationConfig

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("2D DEM Simulation")
        self.setGeometry(100, 100, 1200, 800)

        splitter = QSplitter()
        splitter.setOrientation(Qt.Horizontal)

        self.input_widget = InputWidget()
        self.output_widget = OutputWidget()

        self.simulation_thread = SimulationThread()

        splitter.addWidget(self.input_widget)
        splitter.addWidget(self.output_widget)

        self.setCentralWidget(splitter)

        menubar = QMenuBar(self)
        file_menu = QMenu("Файл", self)
        exit_action = QAction("Выход", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        menubar.addMenu(file_menu)

        toolbar = QToolBar("Основная панель")
        self.addToolBar(toolbar)

        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)

        self.setMenuBar(menubar)

        self.input_widget.apply_button.clicked.connect(self.start_simulation)

        self.resize(1200, 800)

    def start_simulation(self):
        try:
            config = self.input_widget.get_config()
            simulation = Simulation(config)
            self.simulation_thread.setSimulation(simulation)
            self.simulation_thread.finished.connect(self.on_simulation_finished)
            self.simulation_thread.start()
            self.statusBar.showMessage("Симуляция запущена...")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось запустить симуляцию: {str(e)}")

    def on_simulation_finished(self, simulation):
        try:
            trajectories = simulation.get_trajectories()
            self.output_widget.update_particles(trajectories)
            self.output_widget.show_results(simulation)
            self.statusBar.showMessage("Симуляция завершена")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось обработать результаты: {str(e)}")
