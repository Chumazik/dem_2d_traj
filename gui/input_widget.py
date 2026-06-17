from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QFormLayout, QHBoxLayout, QSpinBox, QDoubleSpinBox, QPushButton, QLabel)
from utils.config import SimulationConfig
                                                                                                                                                                                  
class InputWidget(QWidget):                                                                                                                                                      
    """Панель ввода параметров симуляции."""                                                                                                                                     
    def __init__(self):                                                                                                                                                          
        super().__init__()                                                                                                                                                       
        self.layout = QVBoxLayout()                                                                                                                                              
        self.setLayout(self.layout)                                                                                                                                              
                                                                                                                                                                                 
        # ---------- Блок 1 ----------                                                                                                                                           
        block1 = QFormLayout()                                                                                                                                                   
        self.num_particles_spinbox = QSpinBox()                                                                                                                                  
        self.num_particles_spinbox.setRange(1, 10000)                                                                                                                            
        self.num_particles_spinbox.setValue(100)                                                                                                                                 
                                                                                                                                                                                 
        self.particle_radius_spinbox = QDoubleSpinBox()                                                                                                                          
        self.particle_radius_spinbox.setDecimals(4)                                                                                                                              
        self.particle_radius_spinbox.setSingleStep(0.001)                                                                                                                        
        self.particle_radius_spinbox.setValue(0.02)                                                                                                                              
                                                                                                                                                                                 
        self.particle_density_spinbox = QDoubleSpinBox()                                                                                                                         
        self.particle_density_spinbox.setDecimals(1)                                                                                                                             
        self.particle_density_spinbox.setSingleStep(100)                                                                                                                         
        self.particle_density_spinbox.setValue(2500.0)                                                                                                                           
                                                                                                                                                                                 
        block1.addRow(QLabel("Количество частиц:"), self.num_particles_spinbox)                                                                                                  
        block1.addRow(QLabel("Радиус частицы (м):"), self.particle_radius_spinbox)                                                                                               
        block1.addRow(QLabel("Плотность (кг/м³):"), self.particle_density_spinbox)                                                                                               
                                                                                                                                                                                 
        # ---------- Блок 2 ----------                                                                                                                                           
        block2 = QFormLayout()                                                                                                                                                   
        self.kn_spinbox = QDoubleSpinBox()                                                                                                                                       
        self.kn_spinbox.setDecimals(0)                                                                                                                                           
        self.kn_spinbox.setSingleStep(1e4)                                                                                                                                       
        self.kn_spinbox.setValue(1e5)                                                                                                                                            
                                                                                                                                                                                 
        self.restitution_spinbox = QDoubleSpinBox()                                                                                                                              
        self.restitution_spinbox.setDecimals(2)                                                                                                                                  
        self.restitution_spinbox.setSingleStep(0.05)                                                                                                                             
        self.restitution_spinbox.setValue(0.9)                                                                                                                                   
                                                                                                                                                                                 
        self.friction_static_spinbox = QDoubleSpinBox()                                                                                                                          
        self.friction_static_spinbox.setDecimals(2)                                                                                                                              
        self.friction_static_spinbox.setSingleStep(0.05)                                                                                                                         
        self.friction_static_spinbox.setValue(0.5)                                                                                                                               
                                                                                                                                                                                 
        self.friction_dynamic_spinbox = QDoubleSpinBox()                                                                                                                         
        self.friction_dynamic_spinbox.setDecimals(2)                                                                                                                             
        self.friction_dynamic_spinbox.setSingleStep(0.05)                                                                                                                        
        self.friction_dynamic_spinbox.setValue(0.4)                                                                                                                              
                                                                                                                                                                                 
        block2.addRow(QLabel("Жёсткость Kn (N/m):"), self.kn_spinbox)                                                                                                            
        block2.addRow(QLabel("Коэффициент восстановления:"), self.restitution_spinbox)                                                                                           
        block2.addRow(QLabel("Статическое трение μs:"), self.friction_static_spinbox)                                                                                            
        block2.addRow(QLabel("Динамическое трение μd:"), self.friction_dynamic_spinbox)                                                                                          
                                                                                                                                                                                 
        # ---------- Блок 3 ----------                                                                                                                                           
        block3 = QFormLayout()                                                                                                                                                   
        self.rolling_friction_spinbox = QDoubleSpinBox()                                                                                                                         
        self.rolling_friction_spinbox.setDecimals(3)                                                                                                                             
        self.rolling_friction_spinbox.setSingleStep(0.001)                                                                                                                       
        self.rolling_friction_spinbox.setValue(0.01)                                                                                                                             
                                                                                                                                                                                 
        self.drum_radius_spinbox = QDoubleSpinBox()                                                                                                                              
        self.drum_radius_spinbox.setDecimals(3)                                                                                                                                  
        self.drum_radius_spinbox.setSingleStep(0.01)                                                                                                                             
        self.drum_radius_spinbox.setValue(0.5)                                                                                                                                   
                                                                                                                                                                                 
        self.drum_omega_spinbox = QDoubleSpinBox()                                                                                                                               
        self.drum_omega_spinbox.setDecimals(2)                                                                                                                                   
        self.drum_omega_spinbox.setSingleStep(0.1)                                                                                                                               
        self.drum_omega_spinbox.setValue(2.0)                                                                                                                                    
                                                                                                                                                                                 
        self.dt_spinbox = QDoubleSpinBox()                                                                                                                                       
        self.dt_spinbox.setDecimals(7)                                                                                                                                           
        self.dt_spinbox.setSingleStep(1e-6)                                                                                                                                      
        self.dt_spinbox.setValue(1e-5)                                                                                                                                           
                                                                                                                                                                                 
        self.total_time_spinbox = QDoubleSpinBox()                                                                                                                               
        self.total_time_spinbox.setDecimals(2)                                                                                                                                   
        self.total_time_spinbox.setSingleStep(0.5)                                                                                                                               
        self.total_time_spinbox.setValue(5.0)                                                                                                                                    
                                                                                                                                                                                 
        block3.addRow(QLabel("Коэффициент качения μr:"), self.rolling_friction_spinbox)                                                                                          
        block3.addRow(QLabel("Радиус барабана (м):"), self.drum_radius_spinbox)                                                                                                  
        block3.addRow(QLabel("Угловая скорость барабана (рад/с):"), self.drum_omega_spinbox)                                                                                     
        block3.addRow(QLabel("Шаг по времени Δt (с):"), self.dt_spinbox)                                                                                                         
        block3.addRow(QLabel("Общее время (с):"), self.total_time_spinbox)                                                                                                       
                                                                                                                                                                                 
        # ---------- Кнопка ----------                                                                                                                                           
        button_layout = QHBoxLayout()                                                                                                                                            
        self.apply_button = QPushButton("Применить")                                                                                                                             
        button_layout.addWidget(self.apply_button)                                                                                                                               
                                                                                                                                                                                 
        # ---------- Сборка ----------                                                                                                                                           
        self.layout.addLayout(block1)                                                                                                                                            
        self.layout.addLayout(block2)                                                                                                                                            
        self.layout.addLayout(block3)                                                                                                                                            
        self.layout.addLayout(button_layout)                                                                                                                                                          

    def get_config(self) -> SimulationConfig:
        """Создаёт объект SimulationConfig из текущих значений виджетов.
        """
        return SimulationConfig(
            num_particles=self.num_particles_spinbox.value(),                                                                                                                    
            particle_radius=self.particle_radius_spinbox.value(),                                                                                                                
            particle_density=self.particle_density_spinbox.value(),                                                                                                              
            kn=self.kn_spinbox.value(),
            restitution_coeff=self.restitution_spinbox.value(),                                                                                                                  
            friction_static=self.friction_static_spinbox.value(),                                                                                                                
            friction_dynamic=self.friction_dynamic_spinbox.value(),                                                                                                              
            rolling_friction=self.rolling_friction_spinbox.value(),
            drum_radius=self.drum_radius_spinbox.value(),                                                                                                                        
            drum_omega=self.drum_omega_spinbox.value(),                                                                                                                          
            dt=self.dt_spinbox.value(),                                                                                                                                          
            total_time=self.total_time_spinbox.value()
        )