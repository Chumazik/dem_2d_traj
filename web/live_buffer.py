"""Потокобезопасный буфер промежуточных результатов симуляции.

Хранит уже посчитанные точки траекторий частиц, отметки времени и
значения приводного момента. Используется Flask-приложением, чтобы
отдавать клиенту частичные результаты через эндпоинт /partial_results.
"""

import threading


class LiveBuffer:
    """Буфер для накопления промежуточных результатов симуляции."""

    def __init__(self):
        self._lock = threading.Lock()
        self.reset(0)

    def reset(self, num_particles: int) -> None:
        """Сбрасывает буфер и инициализирует списки под num_particles частиц."""
        with self._lock:
            self._trajectories = [[] for _ in range(int(num_particles))]
            self._time = []
            self._torque = []
            self._last_step = 0
            self._progress = 0.0
            self._running = False

    def mark_running(self, running: bool) -> None:
        with self._lock:
            self._running = bool(running)

    def set_progress(self, progress: float) -> None:
        with self._lock:
            self._progress = float(progress)

    def set_last_step(self, step: int) -> None:
        with self._lock:
            self._last_step = int(step)

    def append(self, particles, t: float, torque) -> None:
        """Дописывает текущую позицию каждой частицы и значения t/torque.

        Источник позиции частицы:
          1) ``particle.history[-1]`` (если он есть и непустой);
          2) ``particle.pos`` (текущая позиция).

        Траектории хранятся как список списков [[x, y], ...] на частицу.
        Если число частиц в ``particles`` отличается от размера буфера,
        добавляются/обрезаются недостающие сегменты.
        """
        with self._lock:
            n = len(self._trajectories)
            for i, p in enumerate(particles):
                if i >= n:
                    self._trajectories.append([])
                    n += 1
                pos = None
                hist = getattr(p, "history", None)
                if hist:
                    pos = hist[-1]
                if pos is None:
                    pos = getattr(p, "pos", None)
                if pos is None:
                    continue
                try:
                    self._trajectories[i].append([float(pos[0]), float(pos[1])])
                except Exception:
                    self._trajectories[i].append([float(pos)])

            self._time.append(float(t))
            self._torque.append(float(torque) if torque is not None else 0.0)

    def snapshot(self) -> dict:
        """Возвращает JSON-сериализуемый снимок текущего состояния буфера."""
        with self._lock:
            return {
                "trajectories": [list(traj) for traj in self._trajectories],
                "time": list(self._time),
                "torque_history": list(self._torque),
                "step": self._last_step,
                "progress": self._progress,
                "running": self._running,
            }
