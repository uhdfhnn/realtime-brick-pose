# DUPLO streaming on the Linux desktop

This runner uses new head-camera frames, the measured rectified 960x600 intrinsics,
and bundled DUPLO meshes. Color proposals replace manually entered snapshot boxes:
blue means 2x4, red means 2x2. It supports one of each inside the outlined image
region. These are color heuristics, not a trained brick classifier. Other matching
objects can be selected. Adjust `color_detector.roi_xyxy` and HSV ranges in
`configs/live_duplo.colors.yaml` if the camera view or lighting changes.

On the Linux host, from the existing repository:

```bash
git switch codex/dexmate-vega-camera
git pull --ff-only
docker exec -it brick-pose bash
```

Inside the existing container where snapshot inference already works:

```bash
cd /workspace
export ROBOT_NAME='dm/vg3eb20f25bb-1u'
export ROBOT_CONFIG='vega_1u'
export ROBOT_IP='172.26.161.243'
unset ZENOH_CONFIG
export PYTHONPATH="/workspace/src:/workspace/deps/bop_toolkit_challenge${PYTHONPATH:+:$PYTHONPATH}"
export MEGAPOSE_DATA_DIR=/workspace/local_data
xvfb-run -a python3 -m megapose.scripts.stream_duplo
```

Open http://127.0.0.1:8090 in a browser on the Linux desktop. The existing
container must use `--network host`. No X11 socket mount or new GUI package is
needed for the browser. Xvfb provides the renderer's offscreen display.

For color-box diagnostics without loading MegaPose, stop with Ctrl+C and run:

```bash
python3 -m megapose.scripts.stream_duplo --camera-only
```

The full mode displays an 80-pixel grid, nominal object boxes, XYZ axes and
positions in millimeters in the camera optical frame (x right, y down, z forward).
The grid is an image grid, not a calibrated tabletop grid. `SCORE PASS` only means
the model score passed a threshold; inspect alignment to judge the pose.

Each completed estimate is drawn on exactly its input image. While estimating
the next frame, the browser holds the last result and shows its display age.
It does not paste old poses on newer frames. Camera acquisition keeps only the
latest frame rather than building a queue. Initialization or reacquisition can
take seconds; tracking speed needs measurement on the GPU. `target_hz: 10` is a
rate cap, not a promised inference rate. Browser polling also does not measure
camera or inference FPS.

Results append to `local_data/duplo_stream.jsonl`. Pose translations are meters,
quaternions are xyzw, and timestamps are host receive times, not sensor exposure
timestamps. There is no robot-base transform or robot motion command. The viewer
binds only to loopback; it serves the preview and status, not repository files.

Validation: CPU tests cover moving/disappearing color proposals, ROI filtering,
HTTP preview transport, JSON output and cleanup with simulated camera/estimator.
GPU performance and real pose accuracy require testing on the user's system.
