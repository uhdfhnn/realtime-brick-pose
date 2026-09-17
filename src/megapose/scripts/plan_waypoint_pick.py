"""Generate a sparse pick plan from a frozen observation; never command hardware."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from megapose.waypoint_pick import PinocchioKinematics, joint_segment, pick_targets, solve_waypoints, transform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--observation', type=Path, required=True, help='One complete stream JSON record')
    parser.add_argument('--state', type=Path, required=True, help='URDF joint-name to measured position JSON map')
    parser.add_argument('--label', choices=['duplo_2x2', 'duplo_2x4'], required=True)
    parser.add_argument('--output', type=Path, default=Path('local_data/pick_plan.json'))
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    for field in ('urdf', 'base_frame', 'ee_frame'):
        if not isinstance(cfg.get(field), str) or not cfg[field]:
            raise ValueError(f'Fill {field} with the actual hardware model value')
    transform(cfg['T_ee_tcp'], 'T_ee_tcp')
    transform(cfg['T_object_tcp'][args.label], f'T_object_tcp.{args.label}')
    observation = json.loads(args.observation.read_text())
    if observation.get('pose_frame') != 'camera_optical':
        raise ValueError('Expected a camera_optical stream observation')
    if observation.get('camera_frame_id') != cfg['camera_frame_id']:
        raise ValueError('Observation and calibrated camera frame IDs differ')
    poses = [p for p in observation['poses'] if p['label'] == args.label and p.get('valid') is True]
    if len(poses) != 1:
        raise ValueError('Exactly one score-passing target required (score is not physical validation)')
    pose = poses[0]
    if pose['object_frame']['origin'] != cfg['object_origin']:
        raise ValueError('Grasp calibration and mesh origin differ')
    urdf = (args.config.resolve().parent / cfg['urdf']).resolve()
    model = PinocchioKinematics(urdf, json.loads(args.state.read_text()), cfg['base_frame'], cfg['ee_frame'])
    arm_names = cfg['arm_joint_names']
    if len(arm_names) != 7 or len(set(arm_names)) != 7:
        raise ValueError('Select exactly seven arm joints')
    # Require actual arm names, not torso/head/gripper DOFs masquerading as extra reach.
    if arm_names not in [[f'{side}_arm_j{i}' for i in range(1, 8)] for side in ['L', 'R']]:
        raise ValueError('Use ordered L_arm_j1..7 or R_arm_j1..7')
    active = [model.indices[n] for n in arm_names]
    camera = cfg['camera_transform']
    if camera['mode'] == 'explicit':
        T_base_camera = transform(camera['T_base_camera'], 'T_base_camera')
    elif camera['mode'] == 'urdf':
        T_base_camera = model.frame(model.q, model.frame_id(camera['frame'])) @ transform(
            camera['T_frame_camera'], 'T_frame_camera')
    else:
        raise ValueError('camera_transform.mode must be explicit or urdf')
    T_base_object, targets = pick_targets(T_base_camera, pose['T_camera_object_m'],
        cfg['T_object_tcp'][args.label], cfg['T_ee_tcp'], cfg['approach_axis_base'],
        cfg['pregrasp_m'], cfg['lift_m'])
    points = solve_waypoints(model.fk, model.q, active, model.model.lowerPositionLimit,
                            model.model.upperPositionLimit, targets, **cfg['ik'])
    previous = model.q[active]
    for point in points:
        goal = np.asarray(point['q'])[active]
        point['arm_q_rad'] = goal.tolist()
        point['arm_joint_samples_rad'] = joint_segment(previous, goal, cfg['max_joint_step_rad']).tolist()
        previous = goal
    report = dict(mode='offline_only', hardware_executed=False, collision_checked=False,
        source_observation=observation, source_joint_positions=json.loads(args.state.read_text()),
        urdf_sha256=hashlib.sha256(urdf.read_bytes()).hexdigest(), configuration=cfg,
        arm_joint_names=arm_names, model_joint_names=model.names,
        T_base_camera_m=T_base_camera.tolist(), T_base_object_m=T_base_object.tolist(),
        events=['open_gripper', 'pregrasp', 'grasp', 'close_gripper', 'lift'], waypoints=points,
        notes=['All endpoint FK checks passed; this is not collision or grasp validation.',
               'Joint samples have no timing law and are not SDK motor commands.',
               'Observation/state temporal alignment and real TCP calibration must be verified before execution.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Offline plan saved: {args.output}. No hardware connected or commanded.')


if __name__ == '__main__':
    main()
