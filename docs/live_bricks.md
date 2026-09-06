# Live 2x4 / 2x2 / 1x2 brick poses

This pipeline reads the newest RGB camera frame at 10 Hz, runs a custom lightweight YOLO
detector, and emits one JSON line containing each visible brick's object-to-camera pose.

## Required assets

1. A calibrated RGB camera and its `K` matrix at the configured resolution.
2. Custom YOLO detect weights whose class names are exactly `brick_2x4`, `brick_2x2`, and
   `brick_1x2`. A COCO-pretrained checkpoint cannot distinguish these classes. YOLO11n is the
   recommended training starting point.
3. One accurate `.ply` or `.obj` mesh per class, with vertex coordinates in millimetres and the
   origin/axes documented in the YAML config.
4. Symmetry transforms that are actually valid for those meshes. The example assumes plain,
   centered geometry: 180 degrees around +z for 2x4 and 1x2, and 90-degree steps for 2x2.

Copy and edit the example:

```bash
cp configs/live_bricks.example.yaml configs/live_bricks.yaml
python -m megapose.scripts.run_live_bricks \
  --config configs/live_bricks.yaml \
  --output-jsonl local_data/live_brick_poses.jsonl
```

The first sighting or a low-overlap reacquisition runs MegaPose's expensive coarse orientation
search. On this RTX 5090, one-hypothesis/one-refinement initialization measured about 1.1 seconds.
Normal tracking reuses the previous pose as the refinement seed and measured about 20 ms, leaving
room for YOLO and capture inside the requested 100 ms frame period. Every record includes `mode`,
latencies, and `deadline_missed`; the consumer must not assume initialization meets 10 Hz.
Low-scoring poses are emitted with `valid=false` and are not reused as tracking seeds.

## Output contract

`T_camera_object_m` maps a point from the documented object frame into the camera optical frame.
Translations and `position_m` are metres. `quaternion_xyzw` uses x/y/z/w ordering. The camera
optical convention is +x right, +y down, +z forward when `K` follows OpenCV convention.

For robot/world output, apply the calibrated extrinsic transform outside this module:

```text
T_world_object = T_world_camera @ T_camera_object
```

Do not command a robot from these poses until mesh scale/origin, camera intrinsics, camera-to-robot
extrinsics, timestamp age, and pose error have been validated on the real setup.

## Current scope

The detector intentionally keeps only the highest-confidence instance of each configured class.
This supports one 2x4, one 2x2, and one 1x2 simultaneously. Multiple bricks of the same type need
an instance tracker (and per-instance pose association) before they are safe to manipulate.
