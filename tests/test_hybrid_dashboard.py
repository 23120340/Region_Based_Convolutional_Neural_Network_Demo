import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from assembly.config import load_config
from assembly.fsm import ConfigurableAssemblyTracker
from assembly.hybrid_dashboard import draw_dashboard
from assembly.model_contract import Prediction
from assembly.project_config import load_project_config, build_fusion_engine


class DashboardTests(unittest.TestCase):
    def test_render_keeps_inference_frame_unmodified_at_different_camera_sizes(self):
        project = load_project_config(ROOT / "configs/projects/earbud_v2.json", ROOT)
        fusion = build_fusion_engine(project.fusion)
        tracker = ConfigurableAssemblyTracker(load_config(project.fsm_config))
        for shape in ((480, 640, 3), (720, 1280, 3), (1280, 720, 3)):
            frame = np.full(shape, 120, dtype=np.uint8)
            original = frame.copy()
            screen = draw_dashboard(
                frame, tracker=tracker, fusion=fusion,
                prediction=Prediction("insert_first_earbud", 0.5),
                prediction_fresh=True, action_threshold=0.5,
            )
            self.assertEqual(screen.shape, (900, 1440, 3))
            self.assertTrue(np.array_equal(frame, original))


if __name__ == "__main__":
    unittest.main()
