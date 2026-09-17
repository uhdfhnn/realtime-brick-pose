# Linux live pose preview

Run on the Linux CUDA inference workstation in its graphical desktop session.
The Python environment needs an OpenCV build with HighGUI (GTK/Qt); an exclusively
`opencv-python-headless` installation cannot open this window. Use a desktop or
remote desktop session with a working display. Do not start the Jetson's dexsensor
over `ssh -X`; this preview is on the inference host, separate from sensor startup.

After updating the branch, copy the measured camera profile if you have not already
created a local config. Existing local copies must also receive the measured 960x600
K from the updated example. Retain your actual detector and mesh paths.

```bash
python -m megapose.scripts.run_live_bricks \
  --config configs/live_bricks.dexmate.yaml \
  --preview \
  --output-jsonl local_data/dexmate_brick_poses.jsonl
```

The window shows the same image used for each pose estimate:

- An 80-pixel image grid. This is not a metric table/world grid.
- Detector bounding boxes and nominal centered 3D dimension envelopes projected
  with the calibrated K and `T_camera_object_m`. Envelopes are not CAD mesh renders;
  they are shown only for the documented geometric-center object origin.
- Object XYZ axes: X red, Y green, Z blue. Object origins and dimensions must match
  the actual mesh. Screen labels show camera-frame position in millimetres,
  detection confidence, pose score, and validity. JSONL still uses metres.
- Inference mode and per-frame inference latency. Green denotes valid-score poses;
  orange denotes low-score poses, not validated control targets.

Keys: `g` toggles pixel grid, `a` toggles 3D axes/envelopes, `q` or Escape exits.
Closing the window also exits. YAML `runtime.preview: true` is an alternative to
`--preview`.

The display updates once per completed inference. Coarse initialization and
reacquisition can pause updates; this is not an independent 30-fps video viewer.
A camera timeout terminates the pipeline instead of manufacturing new poses.

This feature does not supply DUPLO assets: the example's object entries still
represent small LEGO bricks. Replace them with the correct 2x2/2x4 DUPLO geometry
and detector before evaluating real DUPLO poses. Camera-only reception can be
checked separately with `megapose.scripts.check_dexmate_camera`.

Validation: 12 CPU tests pass (8 adapter tests plus 4 projection/rendering tests),
including optical projection, rotated axes, invalid depths, image ownership and
overlay toggles. Modified Python files compile. Drawing was tested with headless
OpenCV; a live Linux desktop window, CUDA inference and robot feed have not been
validated in this environment.
