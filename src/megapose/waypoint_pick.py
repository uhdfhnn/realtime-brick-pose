"""Offline sparse pick planning. No vendor SDK imports or robot commands.

Transforms T_A_B map B coordinates into A. Distances are metres, angles radians.
Inspired by BrickBench's sparse target/selected-joint separation, not its simulator
mounting constants, gripper commands, or reactive state machine.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def transform(value, name='transform'):
    T = np.asarray(value, dtype=float)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ValueError(f'{name}: explicit finite 4x4 matrix required')
    R = T[:3, :3]
    if (not np.allclose(T[3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(R.T @ R, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(R), 1, atol=1e-5)):
        raise ValueError(f'{name}: invalid rigid transform')
    return T.copy()


def pick_targets(T_base_camera, T_camera_object, T_object_tcp, T_ee_tcp,
                 approach_axis_base, pregrasp_m, lift_m):
    """Three sparse EE poses; explicit grasp geometry avoids Euler-axis guesses."""
    T_base_object = transform(T_base_camera) @ transform(T_camera_object)
    grasp_tcp = T_base_object @ transform(T_object_tcp, 'T_object_tcp')
    axis = np.asarray(approach_axis_base, dtype=float)
    if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-9:
        raise ValueError('A nonzero approach axis in base coordinates is required')
    axis /= np.linalg.norm(axis)
    if not np.isfinite([pregrasp_m, lift_m]).all() or min(pregrasp_m, lift_m) <= 0:
        raise ValueError('Pregrasp and lift distances must be positive metres')
    tcp_to_ee = np.linalg.inv(transform(T_ee_tcp, 'T_ee_tcp'))
    targets = []
    for name, distance in [('pregrasp', pregrasp_m), ('grasp', 0.), ('lift', lift_m)]:
        tcp = grasp_tcp.copy()
        tcp[:3, 3] += axis * distance
        targets.append((name, tcp @ tcp_to_ee))
    return T_base_object, targets


def solve_waypoints(fk, q_start, active_indices, lower, upper, targets,
                    position_tolerance_m=.002, rotation_tolerance_rad=.035,
                    max_nfev=1000):
    """Bounded endpoint IK, preserving all inactive joints at measured values."""
    q = np.asarray(q_start, dtype=float).copy()
    active = np.asarray(active_indices, dtype=int)
    lo, hi = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if (q.ndim != 1 or not np.isfinite(q).all() or lo.shape != q.shape or hi.shape != q.shape
            or not np.isfinite(lo).all() or not np.isfinite(hi).all()
            or np.any(lo >= hi) or np.any(q < lo) or np.any(q > hi)):
        raise ValueError('Finite in-limit joint state and matching bounds required')
    if (active.ndim != 1 or not len(active) or len(set(active.tolist())) != len(active)
            or np.any(active < 0) or np.any(active >= len(q))):
        raise ValueError('Active joint indices must be unique and in range')
    if (not np.isfinite([position_tolerance_m, rotation_tolerance_rad]).all()
            or min(position_tolerance_m, rotation_tolerance_rad) <= 0):
        raise ValueError('Positive IK acceptance tolerances required')
    results = []
    for name, target in targets:
        target = transform(target)
        fixed = q.copy()
        def residual(x):
            trial = fixed.copy()
            trial[active] = x
            actual = transform(fk(trial))
            return np.r_[(actual[:3, 3] - target[:3, 3]) / position_tolerance_m,
                         Rotation.from_matrix(target[:3, :3].T @ actual[:3, :3]).as_rotvec()
                         / rotation_tolerance_rad]
        fit = least_squares(residual, q[active], bounds=(lo[active], hi[active]),
                            max_nfev=max_nfev, ftol=1e-10, xtol=1e-10, gtol=1e-10)
        err = residual(fit.x)
        p_error = float(np.linalg.norm(err[:3]) * position_tolerance_m)
        r_error = float(np.linalg.norm(err[3:]) * rotation_tolerance_rad)
        if p_error > position_tolerance_m or r_error > rotation_tolerance_rad:
            raise RuntimeError(f'{name}: IK failed FK check ({p_error:.6f} m, {r_error:.6f} rad)')
        q[active] = fit.x
        results.append(dict(name=name, q=q.tolist(), T_base_ee_m=target.tolist(),
                            position_error_m=p_error, rotation_error_rad=r_error))
    return results


def joint_segment(q0, q1, max_step_rad):
    """Offline joint-linear samples, NOT a Cartesian straight-line path."""
    a, b = np.asarray(q0, dtype=float), np.asarray(q1, dtype=float)
    if (a.shape != b.shape or a.ndim != 1 or not np.isfinite(a).all()
            or not np.isfinite(b).all() or not np.isfinite(max_step_rad) or max_step_rad <= 0):
        raise ValueError('Invalid joint interpolation input')
    n = max(1, int(np.ceil(np.max(np.abs(b - a)) / max_step_rad)))
    return np.linspace(a, b, n + 1)


class PinocchioKinematics:
    """Fixed-root, scalar-joint URDF FK. No mesh loading or hardware connection."""
    def __init__(self, urdf, joint_positions, base_frame, ee_frame):
        import pinocchio as pin
        self.pin = pin
        self.model = pin.buildModelFromUrdf(str(urdf))
        self.data = self.model.createData()
        self.names = list(self.model.names)[1:]
        if any(j.nq != 1 or j.nv != 1 for j in list(self.model.joints)[1:]):
            raise ValueError('Only fixed-root URDFs with scalar joints are supported')
        if set(joint_positions) != set(self.names):
            raise ValueError(f'Provide every URDF joint exactly once: {self.names}')
        self.q = np.zeros(self.model.nq)
        self.indices = {}
        for name in self.names:
            idx = self.model.joints[self.model.getJointId(name)].idx_q
            self.indices[name] = idx
            self.q[idx] = float(joint_positions[name])
        self.base = self.frame_id(base_frame)
        self.ee = self.frame_id(ee_frame)

    def frame_id(self, name):
        if not self.model.existFrame(name):
            raise ValueError(f'URDF frame absent: {name}')
        return self.model.getFrameId(name)

    def frame(self, q, frame_id):
        self.pin.forwardKinematics(self.model, self.data, q)
        self.pin.updateFramePlacements(self.model, self.data)
        return (self.data.oMf[self.base].inverse() * self.data.oMf[frame_id]).homogeneous.copy()

    def fk(self, q):
        return self.frame(q, self.ee)
