"""CPU geometry/solver checks; no hardware or Pinocchio runtime required."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np
from scipy.spatial.transform import Rotation

spec = importlib.util.spec_from_file_location('waypoint_pick', Path(__file__).parents[1] / 'src/megapose/waypoint_pick.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class PickTest(unittest.TestCase):
    def test_camera_composition_and_tcp_offset(self):
        camera = np.eye(4)
        camera[:3, :3] = Rotation.from_euler('z', 90, degrees=True).as_matrix()
        camera[:3, 3] = [1, 2, 3]
        obj = np.eye(4); obj[0, 3] = .2
        grasp = np.eye(4); grasp[2, 3] = .01
        tcp = np.eye(4); tcp[2, 3] = .1
        base_obj, points = m.pick_targets(camera, obj, grasp, tcp, [0, 0, 1], .08, .1)
        np.testing.assert_allclose(base_obj[:3, 3], [1, 2.2, 3])
        np.testing.assert_allclose(points[1][1][:3, 3], [1, 2.2, 2.91])
        np.testing.assert_allclose(points[0][1][:3, 3] - points[1][1][:3, 3], [0, 0, .08])

    def test_invalid_rotation_rejected(self):
        T = np.eye(4); T[0, 0] = -1
        with self.assertRaises(ValueError): m.transform(T)

    def test_endpoint_ik_preserves_inactive_joint_context(self):
        # Synthetic six-axis Cartesian mechanism plus one fixed-context variable.
        def fk(q):
            T = np.eye(4)
            T[:3, 3] = q[:3] + [q[6], 0, 0]
            T[:3, :3] = Rotation.from_rotvec(q[3:6]).as_matrix()
            return T
        q = np.zeros(7); q[6] = .3
        desired = q.copy(); desired[:6] = [.2, -.1, .3, .1, .2, .1]
        target = fk(desired)
        results = m.solve_waypoints(fk, q, list(range(6)), -np.ones(7), np.ones(7), [('grasp', target)])
        result = results[0]
        self.assertEqual(result['q'][6], .3)
        np.testing.assert_allclose(fk(np.asarray(result['q'])), target, atol=1e-5)
        bad = target.copy(); bad[0, 3] = 10
        with self.assertRaises(RuntimeError):
            m.solve_waypoints(fk, q, list(range(6)), -np.ones(7), np.ones(7), [('unreachable', bad)])

    def test_joint_samples_bound_steps_and_include_endpoints(self):
        a, b = np.zeros(7), np.array([.1, -.2, 0, .05, 0, 0, 0])
        samples = m.joint_segment(a, b, .01)
        np.testing.assert_allclose(samples[0], a)
        np.testing.assert_allclose(samples[-1], b)
        self.assertLessEqual(np.max(np.abs(np.diff(samples, axis=0))), .01000001)


if __name__ == '__main__': unittest.main()
