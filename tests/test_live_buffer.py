import threading
import unittest

from web.live_buffer import LiveBuffer


class _Particle:
    def __init__(self, x, y):
        self.pos = (x, y)


class TestLiveBuffer(unittest.TestCase):

    def test_full_snapshot_returns_all_samples(self):
        lock = threading.RLock()
        buf = LiveBuffer(lock=lock)
        buf.reset(2)
        particles = [_Particle(0.1, 0.0), _Particle(0.2, 0.0)]
        for k in range(6):
            particles[0].pos = (0.1 + k * 0.01, 0.0)
            particles[1].pos = (0.2 + k * 0.01, 0.005 * k)
            buf.append(particles, k * 0.01, k,
                       running=True, progress=k * 10, last_step=k)
        full = buf.snapshot()
        self.assertEqual(len(full["trajectories"][0]), 6)
        self.assertEqual(len(full["trajectories"][1]), 6)
        self.assertEqual(len(full["time"]), 6)
        self.assertEqual(len(full["torque_history"]), 6)
        self.assertEqual(full["step"], 5)

    def test_tail_snapshot_truncates_history(self):
        lock = threading.RLock()
        buf = LiveBuffer(lock=lock)
        buf.reset(2)
        particles = [_Particle(0.1, 0.0), _Particle(0.2, 0.0)]
        for k in range(6):
            particles[0].pos = (0.1 + k * 0.01, 0.0)
            particles[1].pos = (0.2 + k * 0.01, 0.005 * k)
            buf.append(particles, k * 0.01, k,
                       running=True, progress=k * 10, last_step=k)

        tail = buf.snapshot(tail=2)
        self.assertEqual(len(tail["trajectories"][0]), 2)
        self.assertEqual(len(tail["trajectories"][1]), 2)
        self.assertEqual(len(tail["time"]), 2)
        self.assertEqual(len(tail["torque_history"]), 2)
        self.assertEqual(tail["trajectories"][0], [
            [0.1 + 0.04, 0.0],
            [0.1 + 0.05, 0.0],
        ])
        self.assertEqual(tail["time"], [0.04, 0.05])
        self.assertEqual(tail["torque_history"], [4, 5])
        self.assertEqual(tail["step"], 5)

        tail1 = buf.snapshot(tail=1)
        self.assertEqual(len(tail1["trajectories"][0]), 1)
        self.assertEqual(tail1["trajectories"][0][0], [0.1 + 0.05, 0.0])

    def test_tail_zero_or_negative_returns_full(self):
        lock = threading.RLock()
        buf = LiveBuffer(lock=lock)
        buf.reset(1)
        particles = [_Particle(0.0, 0.0)]
        for k in range(3):
            particles[0].pos = (k * 0.01, 0.0)
            buf.append(particles, k, k, running=True, progress=k, last_step=k)
        full1 = buf.snapshot(tail=0)
        full2 = buf.snapshot(tail=-5)
        self.assertEqual(len(full1["trajectories"][0]), 3)
        self.assertEqual(len(full2["trajectories"][0]), 3)

    def test_snapshot_contains_max_force_and_max_velocity(self):
        lock = threading.RLock()
        buf = LiveBuffer(lock=lock)
        buf.reset(1)
        p = _Particle(0.0, 0.0)
        buf.append([p], 0.01, 0.5,
                    running=True, progress=1.0, last_step=1,
                    max_force=12.5, max_velocity=0.03)
        snap = buf.snapshot()
        self.assertIn("max_force", snap)
        self.assertIn("max_velocity", snap)
        self.assertAlmostEqual(snap["max_force"], 12.5)
        self.assertAlmostEqual(snap["max_velocity"], 0.03)

    def test_max_force_setters_are_safe_under_lock(self):
        import concurrent.futures
        lock = threading.RLock()
        buf = LiveBuffer(lock=lock)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            futures = [ex.submit(buf.set_max_force, k * 1.0) for k in range(50)]
            [f.result(timeout=5) for f in futures]
            futures2 = [ex.submit(buf.set_max_velocity, k * 0.01) for k in range(50)]
            [f.result(timeout=5) for f in futures2]
        # Должно быть установлено какое-то значение (последнее выигрывает).
        self.assertIsInstance(buf.snapshot()["max_force"], float)
        self.assertIsInstance(buf.snapshot()["max_velocity"], float)

    def test_snapshot_independent_under_lock(self):
        """snapshot() не должен захватывать lock навечно при вызове разных потоков."""
        import concurrent.futures
        lock = threading.RLock()
        buf = LiveBuffer(lock=lock)
        buf.reset(3)
        particles = [_Particle(0.0, 0.0) for _ in range(3)]
        for k in range(20):
            for i, p in enumerate(particles):
                p.pos = (i * 0.01 + k * 0.001, k * 0.001)
            buf.append(particles, k, k, running=True, progress=k * 5, last_step=k)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            futures = [ex.submit(buf.snapshot, tail=1) for _ in range(20)]
            results = [f.result(timeout=5) for f in futures]
        for snap in results:
            self.assertEqual(len(snap["trajectories"]), 3)
            self.assertLessEqual(len(snap["trajectories"][0]), 1)


if __name__ == "__main__":
    unittest.main()
