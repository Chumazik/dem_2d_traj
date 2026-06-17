from PyQt5.QtCore import QThread, pyqtSignal
from dem.simulation import Simulation

class SimulationThread(QThread):
    progress = pyqtSignal(int)
    frame_ready = pyqtSignal(list)
    finished = pyqtSignal(object)

    def __init__(self, simulation: Simulation = None):
        super().__init__()
        self.simulation = simulation

    def setSimulation(self, simulation: Simulation):
        self.simulation = simulation

    def run(self):
        if self.simulation is None:
            return                                                                                                                                                               
        try:                                                                                                                                                                     
            self.simulation.run()                                                                                                                                                
            self.finished.emit(self.simulation)                                                                                                                                  
        except Exception as e:                                                                                                                                                   
            print(f"Ошибка в потоке симуляции: {e}")
            
