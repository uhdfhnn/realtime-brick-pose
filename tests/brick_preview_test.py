"""Projection and CPU rendering checks, independent of CUDA and robot SDK."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location(
    'brick_preview', Path(__file__).parents[1] / 'src/megapose/brick_preview.py')
preview = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preview)


class PreviewTest(unittest.TestCase):
    def test_metric_projection(self):
        K = np.array([[400., 0, 480], [0, 400, 300], [0, 0, 1]])
        T = np.eye(4)
        T[2, 3] = 1
        uv, valid = preview.project_points([[0, 0, 0], [.1, 0, 0], [0, .1, 0]], T, K)
        np.testing.assert_allclose(uv, [[480, 300], [520, 300], [480, 340]])
        self.assertTrue(valid.all())

    def test_rotated_object_axes(self):
        T = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]])
        uv, valid = preview.project_points([[.1, 0, 0]], T, np.diag([400, 400, 1]))
        np.testing.assert_allclose(uv, [[0, 40]])
        self.assertTrue(valid.all())

    def test_invalid_depth(self):
        _, valid = preview.project_points([[0, 0, -1], [0, 0, 0], [np.nan, 0, 1]],
                                         np.eye(4), np.eye(3))
        self.assertFalse(valid.any())

    def test_render_is_copy_and_toggles_work(self):
        import cv2
        canvas = np.zeros((600, 960, 3), dtype=np.uint8)
        T = np.eye(4)
        T[:3, 3] = [.03, .01, .4]
        pose = dict(label='synthetic_2x4', valid=True, bbox_xyxy_px=[460, 285, 560, 350],
                    T_camera_object_m=T, mesh_dimensions_mm=[64, 32, 24],
                    object_frame={'origin': 'geometric center of the nominal outer bounding box'},
                    position_m=T[:3, 3], detection_confidence=.9, pose_score=.8)
        K = [[367.4278259277344, 0, 483.36553955078125],
             [0, 367.4278259277344, 326.57806396484375], [0, 0, 1]]
        on = preview.draw_pose_preview(canvas, [pose], K, 'synthetic', 30)
        off = preview.draw_pose_preview(canvas, [pose], K, 'synthetic', 30, grid=False, axes=False)
        self.assertEqual(on.shape, canvas.shape)
        self.assertEqual(canvas.sum(), 0)
        self.assertGreater(np.count_nonzero(on != off), 0)
        self.assertGreater(preview.draw_pose_preview(canvas, [], K, 'no_detection', 0).sum(), 0)
        pose['valid'] = False
        self.assertFalse(np.array_equal(on, preview.draw_pose_preview(canvas, [pose], K, 'synthetic', 30)))


if __name__ == '__main__':
    unittest.main()
