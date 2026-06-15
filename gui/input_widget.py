from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QSpinBox, QDoubleSpinBox, QPushButton, QMessageBox
from utils.config import SimulationConfig

class InputWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()

        self.num_particles_spinbox = QSpinBox()
        self.particle_radius_doublebox = QDoubleSpinBox()
        self.particle_density_doublebox = QDoubleSpinBox()
        self.kn_doublebox = QDoubleSpinBox()
        self.restitution_doublebox = QDoubleSpinBox()
        self.friction_static_doublebox = QDoubleSpinBox()
        self.friction_dynamic_doublebox = QDoubleSpinBox()
        self.rolling_friction_doublebox = QDoubleSpinBox()
        self.drum_radius_doublebox = QDoubleSpinBox()
        self.drum_omega_doublebox = QDoubleSpinBox()
        self.dt_doublebox = QDoubleSpinBox()
        self.total_time_doublebox = QDoubleSpinBox()

        for label, widget in [
            ("Количество частиц (шт)", self.num_particles_spinbox),
            ("Радиус частицы (м)", self.particle_radius_doublebox),
            ("Плотность частицы (кг/м³)", self.particle_density_doublebox),
            ("Нормальная жесткость (kn) (Н/м)", self.kn_doublebox),
            ("Коэффициент восстановления", self.restitution_doublebox),
            ("Статический коэффициент трения", self.friction_static_doublebox),
            ("Динамический коэффициент трения", self.friction_dynamic_doublebox),
            ("Коэффициент сопротивления качению", self.rolling_friction_doublebox),
            ("Радиус барабана (м)", self.drum_radius_doublebox),
            ("Угловая скорость барабана (рад/с)", self.drum_omega_doublebox),
            ("Шаг по времени (dt) (с)", self.dt_doublebox),
            ("Общее время симуляции (с)", self.total_time_doublebox)
        ]:
            label_widget = QLabel(label)
            label_widget.setBuddy(widget)
            self.layout.addWidget(label_widget)
            self.layout.addWidget(widget)

        self.num_particles_spinbox.setValue(100)
        self.particle_radius_doublebox.setValue(0.02)
        self.particle_density_doublebox.setValue(2500)
        self.kn_doublebox.setValue(1e5)
        self.restitution_doublebox.setValue(0.9)
        self.friction_static_doublebox.setValue(0.5)
        self.friction_dynamic_doublebox.setValue(0.4)
        self.rolling_friction_doublebox.setValue(0.01)
        self.drum_radius_doublebox.setValue(0.5)
        self.drum_omega_doublebox.setValue(2.0)
        self.dt_doublebox.setValue(1e-5)
        self.total_time_doublebox.setValue(5.0)

        self.num_particles_spinbox.setRange(1, 10000)
        self.particle_radius_doublebox.setRange(0.001, 1.0)
        self.particle_density_doublebox.setRange(100, 10000)
        self.kn_doublebox.setRange(1000, 10000000)
        self.restitution_doublebox.setRange(0.0, 1.0)
        self.friction_static_doublebox.setRange(0.0, 2.0)
        self.friction_dynamic_doublebox.setRange(0.0, 2.0)
        self.rolling_friction_doublebox.setRange(0.0, 0.5)
        self.drum_radius_doublebox.setRange(0.1, 5.0)
        self.drum_omega_doublebox.setRange(0.1, 20.0)
        self.dt_doublebox.setRange(1e-8, 1e-2)
        self.total_time_doublebox.setRange(0.01, 100.0)

        self.apply_button = QPushButton("Применить")
        self.layout.addWidget(self.apply_button)

        self.setLayout(self.layout)

    def get_config(self):
        return SimulationConfig(
            num_particles=int(self.num_particles_spinbox.value()),
            particle_radius=float(self.particle_radius_doublebox.value()),
            particle_density=float(self.particle_density_doublebox.value()),
            kn=float(self.kn_doublebox.value()),
            restitution=float(self.restitution_doublebox.value()),
            friction_static=float(self.friction_static_doublebox.value()),
            friction_dynamic=float(self.friction_dynamic_doublebox.value()),
            rolling_friction=float(self.rolling_friction_doublebox.value()),
            drum_radius=float(self.drum_radius_doublebox.value()),
            drum_omega=float(self.drum_omega_doublebox.value()),
            dt=float(self.dt_doublebox.value()),
            total_time=float(self.total_time_doublebox.value())
        )
