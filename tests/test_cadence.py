import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from assembly.cadence import CumulativeDeadlineScheduler, EmbeddingTelemetry


class CadenceTests(unittest.TestCase):
    def test_variable_source_rates_follow_ten_fps_grid_without_drift(self):
        for source_fps in (12.49, 15.0, 30.0008):
            with self.subTest(source_fps=source_fps):
                scheduler = CumulativeDeadlineScheduler(10.0)
                sampled = []
                for index in range(round(source_fps * 60)):
                    timestamp = index / source_fps
                    if scheduler.due(timestamp):
                        sampled.append(timestamp)
                actual = (len(sampled)-1) / (sampled[-1]-sampled[0])
                self.assertAlmostEqual(actual, 10.0, delta=0.2)
                self.assertEqual(len(sampled), len(set(sampled)))
                expected_deadline = len(sampled) / 10.0
                self.assertAlmostEqual(scheduler.next_deadline, expected_deadline, places=8)

    def test_late_frame_skips_deadline_without_phase_drift(self):
        scheduler = CumulativeDeadlineScheduler(10.0)
        self.assertTrue(scheduler.due(0.0))
        self.assertFalse(scheduler.due(0.04))
        self.assertTrue(scheduler.due(0.36))
        self.assertAlmostEqual(scheduler.next_deadline, 0.4)
        self.assertTrue(scheduler.due(0.4))
        self.assertAlmostEqual(scheduler.next_deadline, 0.5)

    def test_telemetry_reports_actual_window_duration(self):
        telemetry = EmbeddingTelemetry(16, horizon_s=3.0)
        for index in range(16):
            telemetry.mark(index / 10)
        snapshot = telemetry.snapshot
        self.assertAlmostEqual(snapshot.actual_fps, 10.0)
        self.assertAlmostEqual(snapshot.window_duration_s, 1.5)
        self.assertEqual(snapshot.samples, 16)


if __name__ == "__main__":
    unittest.main()
