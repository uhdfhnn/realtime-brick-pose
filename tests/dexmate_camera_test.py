"""CPU-only adapter checks: python -m unittest discover -s tests -p '*_test.py'."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import threading
import unittest
from unittest.mock import patch

import numpy as np

# Avoid importing the MegaPose GPU/OpenCV package initializer.
spec = importlib.util.spec_from_file_location(
    'dexmate_camera', Path(__file__).parents[1] / 'src/megapose/dexmate_camera.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def config():
    return dict(width=2, height=1, K=[[2, 0, 1], [0, 2, .5], [0, 0, 1]],
                rectified=True, frame_id='head_left_optical', requested_fps=1000,
                read_timeout_s=1, sensor='head_camera', image_key='left_rgb')


class Sensor:
    def __init__(self):
        self.rgb = np.array([[[255, 0, 0], [0, 0, 255]]], dtype=np.uint8)
        self.keys = None
        self.error = None

    def get_obs(self, obs_keys):
        self.keys = obs_keys
        if self.error:
            raise self.error
        return {'left_rgb': self.rgb}


class AdapterTest(unittest.TestCase):
    def setUp(self):
        self.sensor = Sensor()
        self.robot = SimpleNamespace(sensors=SimpleNamespace(head_camera=self.sensor))
        self.camera = module.DexmateLatestFrameCamera(config(), robot=self.robot)

    def tearDown(self):
        self.camera.close()

    def test_color_and_copy(self):
        seq, timestamp, bgr = self.camera.read_latest(-1, 1)
        self.assertEqual(bgr.tolist(), [[[0, 0, 255], [255, 0, 0]]])
        self.assertEqual(self.sensor.keys, ['left_rgb'])
        self.assertGreater(timestamp, 0)
        bgr[:] = 0
        self.assertNotEqual(self.camera.read_latest(-1, 1)[2].sum(), 0)

    def test_cached_frames_timeout_and_new_frame_recovers(self):
        seq, _, _ = self.camera.read_latest(-1, 1)
        with self.assertRaises(TimeoutError):
            self.camera.read_latest(seq, .03)
        self.sensor.rgb = np.zeros((1, 2, 3), dtype=np.uint8)
        newer, _, frame = self.camera.read_latest(seq, 1)
        self.assertGreater(newer, seq)
        self.assertEqual(frame.sum(), 0)

    def test_stale_unconsumed_frame_is_not_returned(self):
        self.camera.read_latest(-1, 1)
        with self.camera._condition:
            self.camera._received_monotonic -= 10
        with self.assertRaises(TimeoutError):
            self.camera.read_latest(-1, .03)

    def test_bad_resolution_and_sdk_errors_propagate(self):
        seq, _, _ = self.camera.read_latest(-1, 1)
        self.sensor.rgb = np.zeros((2, 2, 3), dtype=np.uint8)
        with self.assertRaises(RuntimeError) as context:
            self.camera.read_latest(seq, 1)
        self.assertIsInstance(context.exception.__cause__, ValueError)

    def test_disconnect(self):
        seq, _, _ = self.camera.read_latest(-1, 1)
        self.sensor.error = OSError('disconnected')
        with self.assertRaises(RuntimeError) as context:
            self.camera.read_latest(seq, 1)
        self.assertIsInstance(context.exception.__cause__, OSError)

    def test_close_wakes_waiter(self):
        seq, _, _ = self.camera.read_latest(-1, 1)
        errors = []
        def wait():
            try:
                self.camera.read_latest(seq, 10)
            except RuntimeError as exc:
                errors.append(exc)
        worker = threading.Thread(target=wait)
        worker.start()
        self.camera.close()
        worker.join(1)
        self.assertFalse(worker.is_alive())
        self.assertEqual(len(errors), 1)
        self.camera.close()


class ConfigurationTest(unittest.TestCase):
    def test_missing_or_invalid_calibration(self):
        for field, value in [('K', None), ('K', np.zeros((3, 3))),
                             ('rectified', False), ('requested_fps', 0),
                             ('image_key', 'depth'), ('read_timeout_s', float('nan'))]:
            with self.subTest(field=field, value=value):
                settings = config()
                settings[field] = value
                with self.assertRaises(ValueError):
                    module.validate_dexmate_camera(settings)

    def test_vendor_configuration_and_owned_shutdown(self):
        sensors = {name: SimpleNamespace(enabled=False)
                   for name in ('head_camera', 'base_front_camera')}
        sensor = Sensor()
        shutdown_calls = []
        robot = SimpleNamespace(sensors=SimpleNamespace(head_camera=sensor),
                                shutdown=lambda: shutdown_calls.append(True))
        configs = SimpleNamespace(sensors=sensors)
        constructors = []
        def create_robot(**kwargs):
            constructors.append(kwargs)
            return robot
        modules = {'dexcontrol': SimpleNamespace(), 'dexcontrol.core': SimpleNamespace(),
                   'dexcontrol.core.config': SimpleNamespace(get_robot_config=lambda: configs),
                   'dexcontrol.robot': SimpleNamespace(Robot=create_robot)}
        with patch.dict(sys.modules, modules):
            camera = module.DexmateLatestFrameCamera(config())
            try:
                camera.read_latest(-1, 1)
                self.assertTrue(sensors['head_camera'].enabled)
                self.assertFalse(sensors['base_front_camera'].enabled)
                self.assertEqual(constructors, [{'configs': configs}])
            finally:
                camera.close()
            camera.close()
        self.assertEqual(shutdown_calls, [True])


if __name__ == '__main__':
    unittest.main()
