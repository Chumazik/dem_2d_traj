import threading
from PyQt5.QtCore import QThread, pyqtSignal
from dem.simulation import Simulation

class SimulationThread(QThread):
    progress = pyqtSignal(int)
    frame_ready = pyqtSignal(list)
    finished = pyqtSignal(object)
    progress_percentage = pyqtSignal(float)   # новый сигнал для процентов
    paused = pyqtSignal()
    resumed = pyqtSignal()

    def __init__(self, simulation: Simulation = None):
        super().__init__()
        self.simulation = simulation
        self._pause_event = threading.Event()
        self._pause_event.set()           # по умолчанию не паузим
        self._paused = False

    def setSimulation(self, simulation: Simulation):
        self.simulation = simulation

    def run(self):
        if self.simulation is None:
            return
        try:
            total_steps = int(self.simulation.config.total_time / self.simulation.config.dt)
            step_count = 0
            t = 0.0
            prev_progress = -1.0

            while t < self.simulation.config.total_time and not self.simulation.stop_requested:
                # блокировка при паузе
                self._pause_event.wait()

                self.simulation.step()
                step_count += 1
                t += self.simulation.config.dt

                current_progress = (step_count / total_steps) * 100.0

                # эмитируем каждые 0.1%
                if current_progress - prev_progress >= 0.1:
                    self.progress_percentage.emit(current_progress)
                    prev_progress = current_progress

            self.finished.emit(self.simulation)
        except Exception as e:
            print(f"Ошибка в потоке симуляции: {e}")

    def pause(self):
        self._pause_event.clear()
        self._paused = True
        self.paused.emit()

    def resume(self):
        self._pause_event.set()
        self._paused = False
        self.resumed.emit()

    def toggle_pause(self):
        if self._paused:
            self.resume()
        else:
            self.pause()
