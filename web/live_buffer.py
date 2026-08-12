"""Потокобезопасный буфер промежуточных результатов симуляции.

Хранит уже посчитанные точки траекторий частиц, отметки времени и
значения приводного момента. Используется Flask-приложением, чтобы
отдавать клиенту частичные результаты через эндпоинт /partial_results.

Буфер разделяет единый :class:`threading.RLock` с владельцем состояния
(обычно :class:`web.app.SimState`), чтобы все обновления были атомарны
относительно снимка.
"""

import threading
from typing import Iterable, List, Optional, Sequence


class LiveBuffer:
    """Буфер для накопления промежуточных результатов симуляции."""

    def __init__(self, lock: Optional[threading.RLock] = None) -> None:
        self._lock: threading.RLock = lock if lock is not None else threading.RLock()
        self._owns_lock: bool = lock is None
        self.reset(0)

    def reset(self, num_particles: int) -> None:
        """Сбрасывает буфер и инициализирует списки под num_particles частиц."""
        with self._lock:
            self._trajectories: List[List[List[float]]] = [[] for _ in range(int(num_particles))]
            self._time: List[float] = []
            self._torque: List[float] = []
            self._last_step: int = 0
            self._progress: float = 0.0
            self._running: bool = False

    def mark_running(self, running: bool) -> None:
        with self._lock:
            self._running = bool(running)

    def set_progress(self, progress: float) -> None:
        with self._lock:
            self._progress = float(progress)

    def set_last_step(self, step: int) -> None:
        with self._lock:
            self._last_step = int(step)

    def update_status(self, running: Optional[bool] = None,
                      progress: Optional[float] = None,
                      last_step: Optional[int] = None) -> None:
        """Атомарно обновляет сразу несколько полей состояния буфера."""
        with self._lock:
            if running is not None:
                self._running = bool(running)
            if progress is not None:
                self._progress = float(progress)
            if last_step is not None:
                self._last_step = int(last_step)

    def append(
        self,
        particles: Iterable[object],
        t: float,
        torque: Optional[float],
        running: Optional[bool] = None,
        progress: Optional[float] = None,
        last_step: Optional[int] = None,
    ) -> None:
        """Дописывает текущую позицию каждой частицы и значения t/torque.

        Позиция берётся напрямую из ``particle.pos`` (текущая позиция).

        Траектории хранятся как список списков ``[[x, y], ...]`` на частицу.
        Если число частиц в ``particles`` отличается от размера буфера,
        добавляются недостающие сегменты.

        Дополнительно (опционально) атомарно обновляет поля
        ``running``/``progress``/``last_step`` под тем же локом,
        чтобы избежать второго захвата блокировки.
        """
        with self._lock:
            n = len(self._trajectories)
            for i, p in enumerate(particles):
                if i >= n:
                    self._trajectories.append([])
                    n += 1
                pos = getattr(p, "pos", None)
                if pos is None:
                    continue
                try:
                    self._trajectories[i].append([float(pos[0]), float(pos[1])])
                except Exception:
                    self._trajectories[i].append([float(pos)])

            self._time.append(float(t))
            self._torque.append(float(torque) if torque is not None else 0.0)

            if running is not None:
                self._running = bool(running)
            if progress is not None:
                self._progress = float(progress)
            if last_step is not None:
                self._last_step = int(last_step)

    def snapshot(self, tail: int = 0) -> dict:
        """Возвращает JSON-сериализуемый снимок текущего состояния буфера.

        Параметр ``tail`` (если >0) обрезает каждую траекторию и
        массивы ``time``/``torque_history`` до последних ``tail`` отсчётов.
        Полезно для живой отрисовки, чтобы не передавать всю историю по сети.
        """
        with self._lock:
            if tail and tail > 0:
                t_traj = [list(traj[-tail:]) for traj in self._trajectories]
                t_time = list(self._time[-tail:])
                t_torque = list(self._torque[-tail:])
            else:
                t_traj = [list(traj) for traj in self._trajectories]
                t_time = list(self._time)
                t_torque = list(self._torque)
            return {
                "trajectories": t_traj,
                "time": t_time,
                "torque_history": t_torque,
                "step": self._last_step,
                "progress": self._progress,
                "running": self._running,
            }
