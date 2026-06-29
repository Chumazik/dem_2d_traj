import sys
from PyQt5.QtWidgets import (
    QMainWindow, QSplitter, QApplication, QMenuBar, QMenu, QAction,
    QMessageBox, QToolBar, QStatusBar, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QProgressBar, QLabel
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

        # ---- Прогресс и пауза / сброс ----
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1000)          # 1000 шагов для точности 0.1%
        self.progress_label = QLabel("0.0 %")
        self.pause_button = QPushButton("Пауза")
        self.pause_button.setCheckable(True)
        self.pause_button.clicked.connect(self.toggle_pause)

        self.reset_button = QPushButton("Сброс")
        self.reset_button.clicked.connect(self.reset_simulation)

        progress_layout = QHBoxLayout()
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.progress_label)
        progress_layout.addWidget(self.pause_button)
        progress_layout.addWidget(self.reset_button)

        # ---- Центральный виджет ----
        central = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(splitter)
        layout.addLayout(progress_layout)
        central.setLayout(layout)
        self.setCentralWidget(central)

        self.setWindowTitle("2D DEM Simulation")
        self.resize(1200, 800)

        # ----- Status bar -----
        self.status = QStatusBar()
        self.setStatusBar(self.status)

        # ----- Поток симуляции -----
        self.sim_thread = SimulationThread()
        self.sim_thread.finished.connect(self.on_simulation_finished)
        self.sim_thread.progress_percentage.connect(self.update_progress)
        self.sim_thread.paused.connect(self.on_paused)
        self.sim_thread.resumed.connect(self.on_resumed)

        # ----- Кнопка запуска -----
        self.input_widget.apply_button.clicked.connect(self.start_simulation)

    def start_simulation(self):
        try:
            config = self.input_widget.get_config()
            simulation = Simulation(config)
            self.sim_thread.setSimulation(simulation)
            self.status.showMessage("Запуск симуляции...")
            self.progress_bar.setValue(0)
            self.progress_label.setText("0.0 %")
            self.pause_button.setChecked(False)
            self.pause_button.setText("Пауза")
            self.sim_thread.start()
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось запустить симуляцию:\n{e}")

    def update_progress(self, percent: float):
        # percent от 0 до 100
        int_val = int(percent * 10)   # 0.1% -> 1 единица
        self.progress_bar.setValue(int_val)
        self.progress_label.setText(f"{percent:.1f} %")

    def toggle_pause(self):
        if self.sim_thread.isRunning():
            self.sim_thread.toggle_pause()

    def reset_simulation(self):
        """Останавливает текущую симуляцию и сбрасывает интерфейс."""
        if self.sim_thread.isRunning():
            # Запрашиваем остановку потока
            self.sim_thread.request_stop()
            # Отменяем паузу, если она активна, чтобы поток мог завершиться
            if self.sim_thread._paused:
                self.sim_thread.resume()
            # Ждём завершения (с таймаутом для безопасности)
            if not self.sim_thread.wait(3000):
                # Если не завершился, принудительно завершаем (на случай утечки)
                self.sim_thread.terminate()
                self.sim_thread.wait()

        # Сбрасываем прогресс и состояние
        self.progress_bar.setValue(0)
        self.progress_label.setText("0.0 %")
        self.pause_button.setChecked(False)
        self.pause_button.setText("Пауза")
        self.status.showMessage("Симуляция сброшена")

        # Очищаем графики и результаты в output_widget
        self.output_widget.clear_results()

        # Создаём новый чистый поток для следующего запуска (старый уже завершён)
        self.sim_thread = SimulationThread()
        self.sim_thread.finished.connect(self.on_simulation_finished)
        self.sim_thread.progress_percentage.connect(self.update_progress)
        self.sim_thread.paused.connect(self.on_paused)
        self.sim_thread.resumed.connect(self.on_resumed)

    def on_paused(self):
        self.pause_button.setText("Продолжить")
        self.status.showMessage("Симуляция приостановлена")

    def on_resumed(self):
        self.pause_button.setText("Пауза")
        self.status.showMessage("Симуляция продолжается")

    def on_simulation_finished(self, simulation: Simulation):
        try:
            trajectories = simulation.get_trajectories()
            self.output_widget.update_particles(trajectories)
            self.output_widget.show_results(simulation)
            self.status.showMessage("Симуляция завершена")
            self.pause_button.setChecked(False)
            self.pause_button.setText("Пауза")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось отобразить результаты:\n{e}")
