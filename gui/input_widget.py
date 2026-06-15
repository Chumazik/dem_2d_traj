from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QSpinBox, QDoubleSpinBox, QPushButton
from utils.config import SimulationConfig

class InputWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()

        # Параметры барабана
        self.drum_radius_label = QLabel("Радиус барабана (м)")
        self.drum_radius_spinbox = QDoubleSpinBox()
        self.layout.addWidget(self.drum_radius_label)
        self.layout.addWidget(self.drum_radius_spinbox)

        self.drum_omega_label = QLabel("Угловая скорость барабана (рад/с)")
        self.drum_omega_spinbox = QDoubleSpinBox()
        self.layout.addWidget(self.drum_omega_label)
        self.layout.addWidget(self.drum_omega_spinbox)

        # Характеристики материала
        self.particle_radius_label = QLabel("Радиус частицы (м)")
        self.particle_radius_spinbox = QDoubleSpinBox()
        self.layout.addWidget(self.particle_radius_label)
        self.layout.addWidget(self.particle_radius_spinbox)

        self.particle_density_label = QLabel("Плотность частицы (кг/м³)")
        self.particle_density_spinbox = QDoubleSpinBox()
        self.layout.addWidget(self.particle_density_label)
        self.layout.addWidget(self.particle_density_spinbox)

        # Свойства модели
        self.num_particles_label = QLabel("Количество частиц")
        self.num_particles_spinbox = QSpinBox()
        self.layout.addWidget(self.num_particles_label)
        self.layout.addWidget(self.num_particles_spinbox)

        # Коэффициенты взаимодействия
        self.kn_label = QLabel("Нормальная жесткость (kn) (Н/м)")
        self.kn_spinbox = QDoubleSpinBox()
        self.layout.addWidget(self.kn_label)
        self.layout.addWidget(self.kn_spinbox)

        self.restitution_label = QLabel("Коэффициент восстановления")
        self.restitution_spinbox = QDoubleSpinBox()
        self.layout.addWidget(self.restitution_label)
        self.layout.addWidget(self.restitution_spinbox)

        self.friction_static_label = QLabel("Статический коэффициент трения")
        self.friction_static_spinbox = QDoubleSpinBox()
        self.layout.addWidget(self.friction_static_label)
        self.layout.addWidget(self.friction_static_spinbox)

        self.friction_dynamic_label = QLabel("Динамический коэффициент трения")
        self.friction_dynamic_spinbox = QDoubleSpinBox()
        self.layout.addWidget(self.friction_dynamic_label)
        self.layout.addWidget(self.friction_dynamic_spinbox)

        self.rolling_friction_label = QLabel("Коэффициент сопротивления качению")
        self.rolling_friction_spinbox = QDoubleSpinBox()
        self.layout.addWidget(self.rolling_friction_label)
        self.layout.addWidget(self.rolling_friction_spinbox)

        # Кнопка применения настроек
        self.apply_button = QPushButton("Применить")
        self.layout.addWidget(self.apply_button)

        self.setLayout(self.layout)

    def get_config(self):
        return SimulationConfig(
            num_particles=int(self.num_particles_spinbox.value()),
            particle_radius=float(self.particle_radius_spinbox.value()),
            particle_density=float(self.particle_density_spinbox.value()),
            kn=float(self.kn_spinbox.value()),
            restitution=float(self.restitution_spinbox.value()),
            friction_static=float(self.friction_static_spinbox.value()),
            friction_dynamic=float(self.friction_dynamic_spinbox.value()),
            rolling_friction=float(self.rolling_friction_spinbox.value()),
            drum_radius=float(self.drum_radius_spinbox.value()),
            drum_omega=float(self.drum_omega_spinbox.value()),
            dt=1e-5,  # Шаг по времени фиксированный
            total_time=5.0  # Общее время симуляции фиксированное
        )
