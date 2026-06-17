from PyQt5.QtWidgets import QWidget, QVBoxLayout, QFormLayout, QSpinBox, QDoubleSpinBox, QPushButton, QLabel                                                                      
                                                                                                                                                                                  
                                                                                                                                                                                  
                                                                                                                                                                                  
class InputWidget(QWidget):                                                                                                                                                       
                                                                                                                                                                                  
    def __init__(self):                                                                                                                                                           
                                                                                                                                                                                  
        super().__init__()                                                                                                                                                        
                                                                                                                                                                                  
        self.layout = QVBoxLayout()                                                                                                                                               
                                                                                                                                                                                  
                                                                                                                                                                                  
                                                                                                                                                                                  
        # Parameters block 1                                                                                                                                                      
                                                                                                                                                                                  
        params_block1_layout = QFormLayout()                                                                                                                                      
                                                                                                                                                                                  
        self.num_particles_spinbox = QSpinBox()                                                                                                                                   
                                                                                                                                                                                  
        self.particle_radius_doublespinbox = QDoubleSpinBox()                                                                                                                     
                                                                                                                                                                                  
        self.particle_density_doublespinbox = QDoubleSpinBox()                                                                                                                    
                                                                                                                                                                                  
        params_block1_layout.addRow(QLabel("Number of Particles:"), self.num_particles_spinbox)                                                                                   
                                                                                                                                                                                  
        params_block1_layout.addRow(QLabel("Particle Radius (m):"), self.particle_radius_doublespinbox)                                                                           
                                                                                                                                                                                  
        params_block1_layout.addRow(QLabel("Particle Density (kg/m^3):"), self.particle_density_doublespinbox)                                                                    
                                                                                                                                                                                  
                                                                                                                                                                                  
                                                                                                                                                                                  
        # Parameters block 2                                                                                                                                                      
                                                                                                                                                                                  
        params_block2_layout = QFormLayout()                                                                                                                                      
                                                                                                                                                                                  
        self.kn_doublespinbox = QDoubleSpinBox()                                                                                                                                  
                                                                                                                                                                                  
        self.restitution_doublespinbox = QDoubleSpinBox()                                                                                                                         
                                                                                                                                                                                  
        self.friction_static_doublespinbox = QDoubleSpinBox()                                                                                                                     
                                                                                                                                                                                  
        self.friction_dynamic_doublespinbox = QDoubleSpinBox()                                                                                                                    
                                                                                                                                                                                  
        params_block2_layout.addRow(QLabel("Kn (N/m):"), self.kn_doublespinbox)                                                                                                   
                                                                                                                                                                                  
        params_block2_layout.addRow(QLabel("Restitution:"), self.restitution_doublespinbox)                                                                                       
                                                                                                                                                                                  
        params_block2_layout.addRow(QLabel("Friction Static:"), self.friction_static_doublespinbox)                                                                               
                                                                                                                                                                                  
        params_block2_layout.addRow(QLabel("Friction Dynamic:"), self.friction_dynamic_doublespinbox)                                                                             
                                                                                                                                                                                  
                                                                                                                                                                                  
                                                                                                                                                                                  
        # Parameters block 3                                                                                                                                                      
                                                                                                                                                                                  
        params_block3_layout = QFormLayout()                                                                                                                                      
                                                                                                                                                                                  
        self.rolling_friction_doublespinbox = QDoubleSpinBox()                                                                                                                    
                                                                                                                                                                                  
        self.drum_radius_doublespinbox = QDoubleSpinBox()                                                                                                                         
                                                                                                                                                                                  
        self.drum_omega_doublespinbox = QDoubleSpinBox()                                                                                                                          
                                                                                                                                                                                  
        self.dt_doublespinbox = QDoubleSpinBox()                                                                                                                                  
                                                                                                                                                                                  
        self.total_time_doublespinbox = QDoubleSpinBox()                                                                                                                          
                                                                                                                                                                                  
        params_block3_layout.addRow(QLabel("Rolling Friction Coeff:"), self.rolling_friction_doublespinbox)                                                                       
                                                                                                                                                                                  
        params_block3_layout.addRow(QLabel("Drum Radius (m):"), self.drum_radius_doublespinbox)                                                                                   
                                                                                                                                                                                  
        params_block3_layout.addRow(QLabel("Drum Angular Velocity (rad/s):"), self.drum_omega_doublespinbox)                                                                      
                                                                                                                                                                                  
        params_block3_layout.addRow(QLabel("Time Step (s):"), self.dt_doublespinbox)                                                                                              
                                                                                                                                                                                  
        params_block3_layout.addRow(QLabel("Total Time (s):"), self.total_time_doublespinbox)                                                                                     
                                                                                                                                                                                  
                                                                                                                                                                                  
                                                                                                                                                                                  
        # Buttons                                                                                                                                                                 
                                                                                                                                                                                  
        button_layout = QHBoxLayout()                                                                                                                                             
                                                                                                                                                                                  
        self.apply_button = QPushButton("Apply")                                                                                                                                  
                                                                                                                                                                                  
        button_layout.addWidget(self.apply_button)                                                                                                                                
                                                                                                                                                                                  
                                                                                                                                                                                  
                                                                                                                                                                                  
        # Add layouts to main layout                                                                                                                                              
                                                                                                                                                                                  
        self.layout.addLayout(params_block1_layout)                                                                                                                               
                                                                                                                                                                                  
        self.layout.addLayout(params_block2_layout)                                                                                                                               
                                                                                                                                                                                  
        self.layout.addLayout(params_block3_layout)                                                                                                                               
                                                                                                                                                                                  
        self.layout.addLayout(button_layout)                                                                                                                                      
                                                                                                                                                                                  
                                                                                                                                                                                  
                                                                                                                                                                                  
        self.setLayout(self.layout)                                                                                                                                               
                                                                                                                                                                                  
                                                                                                                                                                                  
                                                                                                                                                                                  
    def apply_config(self):                                                                                                                                                       
                                                                                                                                                                                  
        config = {                                                                                                                                                                
                                                                                                                                                                                  
            "num_particles": self.num_particles_spinbox.value(),                                                                                                                  
                                                                                                                                                                                  
            "particle_radius": self.particle_radius_doublespinbox.value(),                                                                                                        
                                                                                                                                                                                  
            "particle_density": self.particle_density_doublespinbox.value(),                                                                                                      
                                                                                                                                                                                  
            "kn": self.kn_doublespinbox.value(),                                                                                                                                  
                                                                                                                                                                                  
            "restitution_coeff": self.restitution_doublespinbox.value(),                                                                                                          
                                                                                                                                                                                  
            "friction_static": self.friction_static_doublespinbox.value(),                                                                                                        
                                                                                                                                                                                  
            "friction_dynamic": self.friction_dynamic_doublespinbox.value(),                                                                                                      
                                                                                                                                                                                  
            "rolling_friction_coeff": self.rolling_friction_doublespinbox.value(),                                                                                                
                                                                                                                                                                                  
            "drum_radius": self.drum_radius_doublespinbox.value(),                                                                                                                
                                                                                                                                                                                  
            "drum_omega": self.drum_omega_doublespinbox.value(),                                                                                                                  
                                                                                                                                                                                  
            "dt": self.dt_doublespinbox.value(),                                                                                                                                  
                                                                                                                                                                                  
            "total_time": self.total_time_doublespinbox.value()                                                                                                                   
                                                                                                                                                                                  
        }                                                                                                                                                                         
                                                                                                                                                                                  
        return config                                                                                                                                                             
                                                                                                                                                                                  
                                                                                                                                                                                  
                                                                                                                                                                                  
    def on_config_applied(self, config):                                                                                                                                          
                                                                                                                                                                                  
        self.num_particles_spinbox.setValue(config["num_particles"])                                                                                                              
                                                                                                                                                                                  
        self.particle_radius_doublespinbox.setValue(config["particle_radius"])                                                                                                    
                                                                                                                                                                                  
        self.particle_density_doublespinbox.setValue(config["particle_density"])                                                                                                  
                                                                                                                                                                                  
        self.kn_doublespinbox.setValue(config["kn"])                                                                                                                              
                                                                                                                                                                                  
        self.restitution_doublespinbox.setValue(config["restitution_coeff"])                                                                                                      
                                                                                                                                                                                  
        self.friction_static_doublespinbox.setValue(config["friction_static"])                                                                                                    
                                                                                                                                                                                  
        self.friction_dynamic_doublespinbox.setValue(config["friction_dynamic"])                                                                                                  
                                                                                                                                                                                  
        self.rolling_friction_doublespinbox.setValue(config["rolling_friction_coeff"])                                                                                            
                                                                                                                                                                                  
        self.drum_radius_doublespinbox.setValue(config["drum_radius"])                                                                                                            
                                                                                                                                                                                  
        self.drum_omega_doublespinbox.setValue(config["drum_omega"])                                                                                                              
                                                                                                                                                                                  
        self.dt_doublespinbox.setValue(config["dt"])                                                                                                                              
                                                                                                                                                                                  
        self.total_time_doublespinbox.setValue(config["total_time"])     