from PyQt5.QtWidgets import QWidget, QVBoxLayout, QFormLayout, QLabel, QSpinBox, QDoubleSpinBox, QPushButton,
QHBoxLayout



class InputWidget(QWidget):

    def __init__(self):

        super().__init__()

        self.layout = QVBoxLayout()



        # Parameters block

        self.parameters_layout = QFormLayout()

        self.drum_radius_label = QLabel("Радиус барабана (м)")

        self.drum_radius_spinbox = QDoubleSpinBox(value=0.5, minimum=0.1, maximum=2.0)

        self.parameters_layout.addRow(self.drum_radius_label, self.drum_radius_spinbox)



        self.drum_omega_label = QLabel("Угловая скорость барабана (рад/с)")

        self.drum_omega_spinbox = QDoubleSpinBox(value=2.0, minimum=1.0, maximum=5.0)

        self.parameters_layout.addRow(self.drum_omega_label, self.drum_omega_spinbox)



        # Particle properties block

        self.particle_properties_layout = QFormLayout()

        self.particle_radius_label = QLabel("Радиус частицы (м)")

        self.particle_radius_spinbox = QDoubleSpinBox(value=0.02, minimum=0.01, maximum=0.1)

        self.particle_properties_layout.addRow(self.particle_radius_label, self.particle_radius_spinbox)



        self.particle_density_label = QLabel("Плотность частицы (кг/м³)")

        self.particle_density_spinbox = QDoubleSpinBox(value=2500, minimum=1000, maximum=10000)

        self.particle_properties_layout.addRow(self.particle_density_label, self.particle_density_spinbox)



        # Simulation properties block

        self.simulation_properties_layout = QFormLayout()

        self.num_particles_label = QLabel("Количество частиц")

        self.num_particles_spinbox = QSpinBox(value=100, minimum=50, maximum=500)

        self.simulation_properties_layout.addRow(self.num_particles_label, self.num_particles_spinbox)



        self.dt_label = QLabel("Шаг времени (с)")

        self.dt_spinbox = QDoubleSpinBox(value=1e-5, minimum=1e-6, maximum=1e-3)

        self.simulation_properties_layout.addRow(self.dt_label, self.dt_spinbox)



        self.total_time_label = QLabel("Общее время (с)")

        self.total_time_spinbox = QDoubleSpinBox(value=5.0, minimum=1.0, maximum=20.0)

        self.simulation_properties_layout.addRow(self.total_time_label, self.total_time_spinbox)



        # Buttons block

        self.buttons_layout = QHBoxLayout()

        self.apply_button = QPushButton("Применить")

        self.apply_button.clicked.connect(self.apply_config)

        self.buttons_layout.addWidget(self.apply_button)



        # Add all layouts to the main layout

        self.layout.addLayout(self.parameters_layout)

        self.layout.addLayout(self.particle_properties_layout)

        self.layout.addLayout(self.simulation_properties_layout)

        self.layout.addLayout(self.buttons_layout)



        self.setLayout(self.layout)



    def apply_config(self):

        config = SimulationConfig(

            num_particles=int(self.num_particles_spinbox.value()),

            particle_radius=float(self.particle_radius_spinbox.value()),

            particle_density=float(self.particle_density_spinbox.value()),

            kn=1e5,

            restitution=0.9,

            friction_static=0.5,

            friction_dynamic=0.4,

            rolling_friction=0.01,

            drum_radius=float(self.drum_radius_spinbox.value()),

            drum_omega=float(self.drum_omega_spinbox.value()),

            dt=float(self.dt_spinbox.value()),

            total_time=float(self.total_time_spinbox.value())

        )

        self.on_config_applied(config)



    def on_config_applied(self, config):

        print("Configuration applied:", config)