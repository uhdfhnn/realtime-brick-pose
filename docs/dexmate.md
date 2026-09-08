# Dexmate Vega 1U camera adaptation

The `dexmate` backend feeds the existing RGB YOLO + MegaPose pipeline from the
vendor `dexcontrol` SDK. It subscribes to `head_camera.left_rgb` by default and
issues no motion commands. Implementation follows the supplied Vega 1U manual:
`get_robot_config()`, `configs.sensors[name].enabled = True`, `Robot(configs=...)`,
`robot.sensors.head_camera.get_obs(obs_keys=["left_rgb"])`, and `robot.shutdown()`.
The vendor SDK is optional and imported only when opening this backend.

## Setup

Use the existing inference environment on a CUDA workstation, with the vendor
`dexcontrol`/`dextop` installation and the same communication certificate as the
Jetson. Both machines must be on the same reachable LAN. Install the SDK using
Dexmate's distribution for that host; it is not added to the public conda file.
The Jetson captures/streams RGB; the workstation runs the models. No ROS bridge
or local USB video device is needed.

On the Jetson, in a normal terminal (not `ssh -X`):

```bash
dexsensor launch --sensor head_camera
```

Set these in the workstation terminal, substituting the actual installed config
and certificate path:

```bash
export ROBOT_NAME='dm/vg3eb20f25bb-1u'
export ROBOT_CONFIG='vega_1u'
export ZENOH_CONFIG="$HOME/.dexmate/comm/zenoh/<cert>/zenoh_peer_config.json5"
dextop firmware info
```

Use the robot name with a slash. `dextop topic list` alone is not a sufficient
identity check. Select `ROBOT_CONFIG` from `dexbot cfg list` for this robot's
actual end effector. Certificates are not stored in this repository.

## Calibration and assets

```bash
cp configs/live_bricks.dexmate.example.yaml configs/live_bricks.dexmate.yaml
```

Edit the copied file before either command below:

- Replace width/height with the actual stream dimensions. The example's 1280x720
  is not a verified operating mode. The client does not resize or reconfigure the
  camera; `requested_fps` controls polling only.
- Obtain the **selected eye's rectified** intrinsic matrix from calibration at
  that resolution and fill `K`. Confirm the SDK image is rectified, then set
  `rectified: true`. The manual does not document a calibration retrieval API;
  this adapter deliberately does not invent one. Unrectified images require a
  separately validated rectification step and matching K before using this path.
- Keep `left_rgb` for the default stereo eye. If choosing `right_rgb`, supply its
  own K and optical `frame_id`; the left-eye calibration is not interchangeable.
- Fill the same custom brick YOLO weights and millimetre mesh paths required by
  the original pipeline. This change does not train or supply those assets.

Check reception without loading detector/pose models, then run inference:

```bash
python -m megapose.scripts.check_dexmate_camera \
  --config configs/live_bricks.dexmate.yaml --frames 30
python -m megapose.scripts.run_live_bricks \
  --config configs/live_bricks.dexmate.yaml \
  --output-jsonl local_data/dexmate_brick_poses.jsonl
```

The pipeline remains RGB-only: depth streaming is unnecessary. Wrist cameras are
not presumed to exist on this 1U. A different sensor can be selected only if its
SDK object exposes the same RGB observation contract and has its own calibration.

## Output, freshness, and coordinates

The SDK supplies RGB. The adapter copies it to BGR for YOLO/OpenCV; the existing
pipeline converts back to RGB for MegaPose. Images must be uint8 HxWx3 at the
configured resolution, otherwise capture fails instead of silently rescaling K.

One background reader retains only the latest changed image. Byte-identical
images are treated as cached observations and do not advance the sequence. A
stream with no changed image within `read_timeout_s` fails instead of repeatedly
publishing the same pose. This is conservative: genuinely identical static
images also time out, and changed images do not prove low transport latency.

Existing schema version 1 fields remain. Dexmate records add `camera_backend`,
`camera_sensor`, `camera_image_key`, and `camera_frame_id`. The legacy
`capture_timestamp_s` field is **local receipt time**, explicitly marked by
`capture_timestamp_source: "host_receive_time"`. `sensor_capture_timestamp_s`
is null and `sensor_age_known` is false. There is no claim of sensor exposure
synchronization, measured transport age, or 10 Hz performance on this robot.
Consumers requiring exposure timing must add a verified SDK timestamp integration.

`T_camera_object_m` still maps object coordinates into the selected camera's
optical frame (+x right, +y down, +z forward), in metres. `frame_id` is a descriptive
label, not a verified URDF link. To obtain a base-frame pose:

`T_base_object = T_base_camera_optical @ T_camera_object_m`

Measure the optical-to-robot calibration. If the head/torso moves, use FK and joint
states synchronized to exposure; a constant base-to-camera matrix is valid only
while the calibrated configuration is fixed. This change does not manufacture
extrinsics, publish base-frame poses, or add grasp execution.

## Validation status

Run the CPU-only adapter tests (NumPy and Python standard library):

```bash
python -m unittest discover -s tests -p '*_test.py' -v
```

Eight tests cover color/copy semantics, cached frames/recovery, stale unconsumed
frames, invalid resolution, disconnect propagation, shutdown wakeup, calibration
validation, and vendor configuration/owned shutdown. These use a fake SDK.
The existing pytest suite could not run in the development environment because
pytest/OpenCV and the full MegaPose runtime are absent. No real robot, trained
weights, calibrated camera, or CUDA inference was available for end-to-end testing.
Verify stream mode, color, K, known brick poses, startup/reacquisition latency, and
frame freshness on the physical setup before using estimates for manipulation.
