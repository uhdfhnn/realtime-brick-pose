# Sparse waypoint pick: offline first

The planner consumes one frozen streaming record and measured joint positions.
It produces three endpoint IK solutions: pregrasp, grasp and lift. The event order
is open gripper, pregrasp, grasp, close gripper, lift. There is no reactive task
machine, placement planner, recovery logic, or dense Cartesian interpolation.
All waypoints must pass physical position and rotation FK tolerances before a
plan is written. Selected-arm joint interpolation is provided for inspection;
it is not a time-parameterized or collision-checked trajectory.

Reference inspected: intelligent-control-lab/BrickBench-Dev, michael-dexmate,
commit e1a16184886275056452c85f9ea1cf63046dafe6. The current DexMate expert uses
`dexmate_cartesian_obs` by default. This implementation follows only its sparse
target/active-joint principle, using bounded least-squares endpoint IK. It does
not import the large policy or reproduce its ordinary/bounded fallback pipeline.
The reference uses tip_l/tip_r, seven selected arm joints and simulator gripper
positions. None of its gripper commands, nominal fixed-waist installation
coordinates (0.200 m / 0.500 rad), or LEGO grasp offsets are hardware calibration.

## Coordinate contract

Use the stream's `T_camera_object_m`, not manually reconstructed RPY:

```
T_base_object = T_base_camera @ T_camera_object
T_base_tcp = T_base_object @ T_object_tcp
T_base_ee = T_base_tcp @ inverse(T_ee_tcp)
```

Each T_A_B maps B coordinates into A. Distances are metres; rotations are proper
3x3 rotation matrices. Grasp calibration must use the bundled mesh origin (full
nominal bounding-box center including studs), not its bottom face. For a tabletop
grasp, calibrate jaw direction and contact height separately for 2x2 and 2x4.
Square-brick quarter-turn ambiguity and rectangular half-turn ambiguity affect
wrist choice; this initial planner uses the selected vision orientation as given.

The camera transform can be an explicit measured base-to-optical transform or
URDF FK at the image's joint state followed by an explicit camera-frame correction.
Do not assume the depth frame and rectified left optical frame coincide. Use
measured head angles; the configured head target is not a timestamped measurement.

## Inputs still required from hardware

- Correct hardware URDF with actual fixed installation transforms and EE frame.
- Measured scalar joint positions for every movable URDF joint, mapped by name.
  Only the seven selected arm joints can move in IK; all others remain fixed.
- Calibrated camera optical transform, EE-to-TCP transform, and object-to-TCP
  grasp transforms. No unknown transform defaults to identity.
- Before an execution adapter can be added: verified joint order/limits and
  gripper open/close interface. Simulation finger angles are not CAN motor values.

## Running

Inside the existing inference container, with its PYTHONPATH configured:

```bash
cp configs/waypoint_pick.example.yaml local_data/waypoint_pick.yaml
# Fill its nulls with measured values. Use an absolute URDF path, or a path
# relative to this YAML. Save one complete streaming record to observation.json
# and a joint-name -> position map to joint_state.json, from the same static pose.
python3 -m megapose.scripts.plan_waypoint_pick \
  --config local_data/waypoint_pick.yaml \
  --observation local_data/observation.json \
  --state local_data/joint_state.json \
  --label duplo_2x4 --output local_data/pick_plan.json
```

This command never constructs Robot(), moves the head, opens a connection or
commands a gripper. It accepts recorded data, so it cannot certify freshness or
time alignment. `valid` in the input is only a pose-score filter. The output
explicitly records that no collision checks or physical execution occurred.
Joint-linear motion between endpoints can bow toward the table; inspect the
entire path before adding a timed executor. Successful endpoint IK alone does
not prove collision clearance, finger fit, or grasp success.

CPU validation covers frame composition, TCP inversion, fixed-joint preservation,
unreachable target rejection and joint sample bounds with a synthetic mechanism.
The Pinocchio hardware-URDF path and real robot execution require the hardware
inputs above and have not been validated in this workspace.
