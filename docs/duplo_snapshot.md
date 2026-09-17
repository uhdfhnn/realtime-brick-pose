# First DUPLO snapshot inspection (no YOLO, no robot motion)

This path uses the saved image, calibrated rectified K, two bundled nominal
DUPLO meshes and manual image boxes. It runs on the already-working Ubuntu 22.04
`brick-pose` container with Python 3.10, torch 2.8.0/cu128 and torchvision 0.23.0.
The original Ubuntu 20.04 Dockerfile and Python 3.9 conda profile are not used.

On the desktop host, in the repository:

```bash
git pull --ff-only origin codex/dexmate-vega-camera
git submodule update --init --recursive
docker exec -it brick-pose bash
```

Inside that container:

```bash
cd /workspace
apt-get update
apt-get install -y rclone xvfb xauth libgl1-mesa-dri libglx-mesa0 libegl1 libgomp1
python3 -m pip install -r requirements-dexmate-inspection.txt
export PYTHONPATH=/workspace/src:/workspace/deps/bop_toolkit_challenge
export MEGAPOSE_DATA_DIR=/workspace/local_data
python3 -c "import pinocchio; from megapose.utils.load_model import load_named_model; print('MegaPose imports OK')"
```

Do not replace the already verified PyTorch install. Dependencies are pinned where
needed for the legacy renderer/Pinocchio/NumPy interfaces. The repository config
now uses sys.executable rather than requiring a fake CONDA_PREFIX. Legacy released
checkpoint loading explicitly uses weights_only=False on PyTorch 2.8; only use
trusted official MegaPose checkpoints/configs from the upstream download below.

Download model files using the upstream repository's rclone configuration:

```bash
python3 -m megapose.scripts.download --megapose_models
```

Confirm both RGB model directories contain config.yaml and checkpoint.pth.tar:

```bash
ls local_data/megapose-models/{coarse-rgb-906902141,refiner-rgb-653307694}/{config.yaml,checkpoint.pth.tar}
```

Run the boxes below only on the original shared 960x600 snapshot: blue 2x4 at the
front center and red/orange 2x2 farther right. Boxes are explicit manual inputs,
not detections. If the image or objects change, replace the coordinates. Running
with --prepare-only writes detections.png without loading model dependencies.

```bash
xvfb-run -a python3 -m megapose.scripts.inspect_duplo_snapshot \
  --image local_data/head_snapshot.png \
  --config configs/live_bricks.dexmate.yaml \
  --box duplo_2x4 453 413 489 437 \
  --box duplo_2x2 558 394 580 414
```

Xvfb permits the first offscreen Panda3D test in the existing container, without
recreating it for desktop display mounts. Mesa may perform rendering in software;
PyTorch pose inference still uses CUDA. This is a correctness test, not a claimed
real-time benchmark. Do not set LIBGL_ALWAYS_SOFTWARE unless diagnosing rendering.

Outputs under local_data/duplo_inspection:
- detections.png: inspect the supplied manual boxes.
- poses.json: object-to-left-optical 4x4 matrices (translation in metres), xyzw
  quaternions, K, resolution, and model scores. Manual detection confidence is 1.0
  by convention, not a measured confidence. Validity threshold 0.5 is provisional.
- mesh_overlay.png: projected mesh silhouettes/contours over RGB (not shaded CAD).
- pose_overlay.png: pixel grid, metric bounding envelopes and object-frame axes.

Open outputs from the Linux host with xdg-open. No robot connection is necessary
for this saved-image run. No base-frame pose or control command is generated.
See assets/duplo/README.md for geometry sources, attribution, nominal dimensions,
origin and symmetry limitations. The small bricks in this snapshot may yield poor
poses; evaluate alignment instead of assuming a high score implies accurate pose.

Validation performed during preparation: recursive CAD expansion and dimensional
checks; visual inspection of both generated meshes; --prepare-only run on the
user's image and visual check of the two boxes; Python compilation. The 12 existing
CPU adapter/projection tests pass. Full model loading/rendering/inference remains
to be verified on the user's RTX 4080; this environment has no CUDA runtime.
