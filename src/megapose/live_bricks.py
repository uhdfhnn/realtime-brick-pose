"""Online RGB detector + MegaPose tracking for a small, known brick inventory."""

from __future__ import annotations

# Standard Library
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, TextIO, Tuple, Union

# Third Party
import cv2
import numpy as np
import pandas as pd
import torch
import yaml

# MegaPose
from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset
from megapose.brick_preview import draw_pose_preview
from megapose.dexmate_camera import DexmateLatestFrameCamera, validate_dexmate_camera
from megapose.inference.types import DetectionsType, ObservationTensor, PoseEstimatesType
from megapose.lib3d.symmetries import DiscreteSymmetry
from megapose.lib3d.transform import Transform
from megapose.utils.load_model import load_named_model
from megapose.utils.tensor_collection import PandasTensorCollection


@dataclass(frozen=True)
class Detection:
    """One class-labelled detector result in pixel coordinates."""

    label: str
    bbox_xyxy: np.ndarray
    confidence: float


@dataclass(frozen=True)
class BrickDefinition:
    """Constant geometry and object-frame metadata for one brick type."""

    label: str
    yolo_class_name: str
    mesh_path: Path
    dimensions_mm: Tuple[float, float, float]
    origin: str
    axes: Mapping[str, str]
    discrete_symmetries: Tuple[np.ndarray, ...]


@dataclass
class TrackedPose:
    """The previous pose used to turn expensive coarse estimation into tracking."""

    bbox_xyxy: np.ndarray
    pose: np.ndarray


def _required(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing required config field {context}.{key}")
    return mapping[key]


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load a YAML config and resolve file paths relative to that YAML file."""

    data = yaml.safe_load(config_path.read_text())
    if not isinstance(data, dict):
        raise ValueError("The live-brick config must contain a YAML mapping")

    base_dir = config_path.resolve().parent
    detector = _required(data, "detector", "root")
    detector["weights"] = str((base_dir / _required(detector, "weights", "detector")).resolve())

    objects = _required(data, "objects", "root")
    if not objects:
        raise ValueError("At least one brick object must be configured")
    for index, obj in enumerate(objects):
        obj["mesh_path"] = str(
            (base_dir / _required(obj, "mesh_path", f"objects[{index}]")).resolve()
        )
    return data


def parse_brick_definitions(config: Mapping[str, Any]) -> List[BrickDefinition]:
    """Parse objects and reject label/class collisions before model startup."""

    definitions: List[BrickDefinition] = []
    for index, item in enumerate(_required(config, "objects", "root")):
        context = f"objects[{index}]"
        transforms = []
        for transform in item.get("symmetry", {}).get("discrete_transforms", []):
            matrix = np.asarray(transform, dtype=np.float64)
            if matrix.shape != (4, 4):
                raise ValueError(f"{context} symmetry transforms must be 4x4 matrices")
            transforms.append(matrix)
        frame = _required(item, "object_frame", context)
        dimensions = tuple(float(value) for value in _required(item, "dimensions_mm", context))
        if len(dimensions) != 3 or any(value <= 0 for value in dimensions):
            raise ValueError(f"{context}.dimensions_mm must contain three positive lengths")
        definition = BrickDefinition(
            label=str(_required(item, "label", context)),
            yolo_class_name=str(_required(item, "yolo_class_name", context)),
            mesh_path=Path(_required(item, "mesh_path", context)),
            dimensions_mm=dimensions,
            origin=str(_required(frame, "origin", f"{context}.object_frame")),
            axes=dict(_required(frame, "axes", f"{context}.object_frame")),
            discrete_symmetries=tuple(transforms),
        )
        definitions.append(definition)

    labels = [definition.label for definition in definitions]
    class_names = [definition.yolo_class_name for definition in definitions]
    if len(labels) != len(set(labels)):
        raise ValueError("Brick labels must be unique")
    if len(class_names) != len(set(class_names)):
        raise ValueError("YOLO class names must be unique")
    return definitions


def validate_files(config: Mapping[str, Any], definitions: List[BrickDefinition]) -> None:
    """Fail early when custom detector weights or one of the metric meshes is absent."""

    weights = Path(config["detector"]["weights"])
    if not weights.is_file():
        raise FileNotFoundError(
            f"Custom YOLO weights not found: {weights}. The checkpoint must be trained with "
            "the configured brick class names; a COCO checkpoint cannot distinguish them."
        )
    for definition in definitions:
        if not definition.mesh_path.is_file():
            raise FileNotFoundError(
                f"Mesh for {definition.label} not found: {definition.mesh_path}"
            )


def make_object_dataset(definitions: List[BrickDefinition]) -> RigidObjectDataset:
    """Create the metric object inventory loaded once by MegaPose."""

    objects = []
    for definition in definitions:
        symmetries = [
            DiscreteSymmetry(transform.copy()) for transform in definition.discrete_symmetries
        ]
        objects.append(
            RigidObject(
                label=definition.label,
                mesh_path=definition.mesh_path,
                mesh_units="mm",
                symmetries_discrete=symmetries,
            )
        )
    return RigidObjectDataset(objects)


class LatestFrameCamera:
    """Continuously drain an OpenCV source so inference always receives its newest frame."""

    def __init__(self, source: Union[int, str], width: int, height: int, requested_fps: float):
        self._capture = cv2.VideoCapture(source)
        if not self._capture.isOpened():
            raise RuntimeError(f"Could not open camera/video source {source!r}")
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._capture.set(cv2.CAP_PROP_FPS, requested_fps)
        # One frame is the smallest OpenCV buffer request. It minimizes stale-frame latency;
        # some camera backends ignore it, which is why the background drain thread is also used.
        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._condition = threading.Condition()
        self._frame: Optional[np.ndarray] = None
        self._capture_timestamp_s: Optional[float] = None
        self._sequence = -1
        self._error: Optional[str] = None
        self._stopping = False
        self._thread = threading.Thread(target=self._reader, name="brick-camera", daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        while not self._stopping:
            ok, frame = self._capture.read()
            if not ok:
                with self._condition:
                    self._error = "Camera read failed"
                    self._condition.notify_all()
                return
            with self._condition:
                self._frame = frame
                self._capture_timestamp_s = time.time()
                self._sequence += 1
                self._condition.notify_all()

    def read_latest(self, after_sequence: int, timeout_s: float) -> Tuple[int, float, np.ndarray]:
        """Wait for and copy a frame newer than ``after_sequence``."""

        deadline = time.monotonic() + timeout_s
        with self._condition:
            while self._sequence <= after_sequence and self._error is None:
                remaining_s = deadline - time.monotonic()
                if remaining_s <= 0:
                    raise TimeoutError(f"No new camera frame arrived within {timeout_s:.3f} s")
                self._condition.wait(remaining_s)
            if self._error is not None:
                raise RuntimeError(self._error)
            assert self._frame is not None and self._capture_timestamp_s is not None
            return self._sequence, self._capture_timestamp_s, self._frame.copy()

    def close(self) -> None:
        self._stopping = True
        self._capture.release()
        # One second bounds shutdown latency while still exceeding normal USB frame periods by an
        # order of magnitude at the requested 30 fps. If the backend cannot unblock read() within
        # this interval the thread is daemonized and process exit remains possible; this fixed
        # safety bound is independent of perception accuracy and need not vary per deployment.
        self._thread.join(timeout=1.0)


class YoloBrickDetector:
    """Ultralytics detector restricted to one highest-confidence instance per brick class."""

    def __init__(self, detector_config: Mapping[str, Any], definitions: List[BrickDefinition]):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is not installed. Recreate/update the megapose environment from "
                "conda/environment_inference.yaml."
            ) from exc

        self._model = YOLO(detector_config["weights"])
        self._confidence = float(_required(detector_config, "confidence", "detector"))
        self._iou = float(_required(detector_config, "nms_iou", "detector"))
        self._image_size = int(_required(detector_config, "image_size", "detector"))
        self._device = str(_required(detector_config, "device", "detector"))
        self._half = bool(_required(detector_config, "half", "detector"))
        self._class_to_label = {
            definition.yolo_class_name: definition.label for definition in definitions
        }

        raw_model_names = self._model.names
        model_names = set(
            raw_model_names.values() if isinstance(raw_model_names, dict) else raw_model_names
        )
        missing = set(self._class_to_label) - model_names
        if missing:
            raise ValueError(
                "YOLO checkpoint is missing configured classes: " + ", ".join(sorted(missing))
            )

    def detect(self, bgr: np.ndarray) -> List[Detection]:
        results = self._model.predict(
            source=bgr,
            conf=self._confidence,
            iou=self._iou,
            imgsz=self._image_size,
            device=self._device,
            half=self._half,
            verbose=False,
        )
        best_by_label: Dict[str, Detection] = {}
        boxes = results[0].boxes
        if boxes is None:
            return []
        for bbox, confidence, class_id in zip(boxes.xyxy, boxes.conf, boxes.cls):
            class_name = results[0].names[int(class_id.item())]
            label = self._class_to_label.get(class_name)
            if label is None:
                continue
            detection = Detection(
                label=label,
                bbox_xyxy=bbox.detach().cpu().numpy().astype(np.float32),
                confidence=float(confidence.item()),
            )
            previous = best_by_label.get(label)
            if previous is None or detection.confidence > previous.confidence:
                best_by_label[label] = detection
        return sorted(best_by_label.values(), key=lambda detection: detection.label)


def detections_to_tensor(detections: List[Detection]) -> DetectionsType:
    """Convert YOLO results to MegaPose's single-image detection collection."""

    # batch_im_id=0 is MegaPose's zero-based index for the only image in this call.
    # It is fixed by the API, dimensionless, and would address the wrong image if changed.
    infos = pd.DataFrame(
        {
            "label": [detection.label for detection in detections],
            "batch_im_id": [0] * len(detections),
            "instance_id": list(range(len(detections))),
            "score": [detection.confidence for detection in detections],
        }
    )
    bboxes = torch.as_tensor(np.stack([detection.bbox_xyxy for detection in detections]))
    return PandasTensorCollection(infos=infos, bboxes=bboxes).cuda()


def bbox_iou(left: np.ndarray, right: np.ndarray) -> float:
    """Compute intersection over union for two xyxy pixel boxes."""

    top_left = np.maximum(left[:2], right[:2])
    bottom_right = np.minimum(left[2:], right[2:])
    intersection = float(np.prod(np.maximum(bottom_right - top_left, 0.0)))
    left_area = float(np.prod(np.maximum(left[2:] - left[:2], 0.0)))
    right_area = float(np.prod(np.maximum(right[2:] - right[:2], 0.0)))
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _rotation_distance(left: np.ndarray, right: np.ndarray) -> float:
    relative = left[:3, :3].T @ right[:3, :3]
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cosine))


def canonicalize_pose(
    pose: np.ndarray, symmetry_poses: np.ndarray, previous_pose: Optional[np.ndarray]
) -> Tuple[np.ndarray, int]:
    """Select the equivalent orientation nearest the previous pose (identity initially)."""

    candidates = [pose @ symmetry_pose for symmetry_pose in symmetry_poses]
    reference = np.eye(4, dtype=np.float64) if previous_pose is None else previous_pose
    distances = [_rotation_distance(reference, candidate) for candidate in candidates]
    symmetry_index = int(np.argmin(distances))
    return candidates[symmetry_index], symmetry_index


class LiveBrickPoseEstimator:
    """Full initialize on acquisition, then fast single-step pose tracking."""

    def __init__(
        self,
        definitions: List[BrickDefinition],
        tracker_config: Mapping[str, Any],
    ):
        self._definitions = {definition.label: definition for definition in definitions}
        self._objects = make_object_dataset(definitions)
        self._symmetry_poses = {
            definition.label: self._objects.get_object_by_label(
                definition.label
            ).make_symmetry_poses()
            for definition in definitions
        }
        self._minimum_iou = float(_required(tracker_config, "minimum_iou", "tracker"))
        if not 0.0 <= self._minimum_iou <= 1.0:
            raise ValueError("tracker.minimum_iou must be in the dimensionless range [0, 1]")
        self._refiner_iterations = int(_required(tracker_config, "refiner_iterations", "tracker"))
        if self._refiner_iterations < 1:
            raise ValueError("tracker.refiner_iterations must be at least one")
        self._minimum_pose_score = float(_required(tracker_config, "minimum_pose_score", "tracker"))
        if not 0.0 <= self._minimum_pose_score <= 1.0:
            raise ValueError("tracker.minimum_pose_score must be in the range [0, 1]")
        # This exact RGB checkpoint is the lightweight single-hypothesis MegaPose release. The
        # task supplies RGB only, so RGB-D/ICP would be invalid; the multi-hypothesis variant would
        # repeat refinement and miss the 100 ms tracking budget. Keep this fixed for the live
        # module and create a separate calibrated profile if depth is later added.
        self._pose_estimator = load_named_model(
            "megapose-1.0-RGB",
            self._objects,
            n_workers=int(_required(tracker_config, "renderer_workers", "tracker")),
            bsz_images=int(_required(tracker_config, "coarse_batch_size", "tracker")),
        ).cuda()
        self._tracked: Dict[str, TrackedPose] = {}

    def close(self) -> None:
        """Stop the shared Panda3D renderer process cleanly."""

        renderers = {
            id(model.renderer): model.renderer
            for model in (self._pose_estimator.coarse_model, self._pose_estimator.refiner_model)
        }
        for renderer in renderers.values():
            renderer.stop()

    def _can_track(self, detections: List[Detection]) -> bool:
        if not detections:
            return False
        return all(
            detection.label in self._tracked
            and bbox_iou(detection.bbox_xyxy, self._tracked[detection.label].bbox_xyxy)
            >= self._minimum_iou
            for detection in detections
        )

    def _make_coarse_estimates(self, detections: List[Detection]) -> PoseEstimatesType:
        infos = pd.DataFrame(
            {
                "label": [detection.label for detection in detections],
                "batch_im_id": [0] * len(detections),
                "instance_id": list(range(len(detections))),
            }
        )
        poses = torch.as_tensor(
            np.stack([self._tracked[detection.label].pose for detection in detections]),
            dtype=torch.float32,
            device="cuda",
        )
        return PandasTensorCollection(infos=infos, poses=poses)

    def estimate(
        self, rgb: np.ndarray, camera_matrix: np.ndarray, detections: List[Detection]
    ) -> Tuple[List[Dict[str, Any]], str, float]:
        """Estimate current poses and return serialized results, mode, and latency."""

        if not detections:
            self._tracked.clear()
            return [], "no_detection", 0.0

        observation = ObservationTensor.from_numpy(rgb=rgb, K=camera_matrix).cuda()
        start = time.perf_counter()
        if self._can_track(detections):
            mode = "track"
            estimates, extra = self._pose_estimator.run_inference_pipeline(
                observation,
                coarse_estimates=self._make_coarse_estimates(detections),
                n_refiner_iterations=self._refiner_iterations,
                # One hypothesis is the upstream RGB model's low-latency setting. Additional
                # hypotheses duplicate refinement work and cannot fit the measured 100 ms period;
                # use a separate quality-first profile if this trade-off changes.
                n_pose_hypotheses=1,
            )
        else:
            mode = "initialize"
            estimates, extra = self._pose_estimator.run_inference_pipeline(
                observation,
                detections=detections_to_tensor(detections),
                n_refiner_iterations=self._refiner_iterations,
                # The single-hypothesis initialization measured roughly 1.1--1.5 seconds here;
                # additional hypotheses worsen reacquisition latency and are inappropriate for
                # this 10 Hz profile, so the setting is intentionally fixed.
                n_pose_hypotheses=1,
            )
        latency_ms = (time.perf_counter() - start) * 1000.0

        detection_by_label = {detection.label: detection for detection in detections}
        output = []
        next_tracked: Dict[str, TrackedPose] = {}
        for row_index, row in estimates.infos.reset_index(drop=True).iterrows():
            label = str(row["label"])
            raw_pose = estimates.poses[row_index].detach().cpu().numpy()
            previous_pose = self._tracked[label].pose if label in self._tracked else None
            pose, symmetry_index = canonicalize_pose(
                raw_pose, self._symmetry_poses[label], previous_pose
            )
            detection = detection_by_label[label]
            transform = Transform(pose)
            pose_score = float(row["pose_score"])
            pose_valid = pose_score >= self._minimum_pose_score
            if pose_valid:
                next_tracked[label] = TrackedPose(detection.bbox_xyxy.copy(), pose.copy())
            definition = self._definitions[label]
            output.append(
                {
                    "label": label,
                    "bbox_xyxy_px": detection.bbox_xyxy.tolist(),
                    "detection_confidence": detection.confidence,
                    "pose_score": pose_score,
                    "valid": pose_valid,
                    "T_camera_object_m": pose.tolist(),
                    "position_m": transform.translation.tolist(),
                    "quaternion_xyzw": transform.quaternion.coeffs().tolist(),
                    "symmetry_index": symmetry_index,
                    "mesh_path": str(definition.mesh_path),
                    "mesh_dimensions_mm": list(definition.dimensions_mm),
                    "object_frame": {"origin": definition.origin, "axes": dict(definition.axes)},
                }
            )
        self._tracked = next_tracked
        return output, mode, float(extra["time"] * 1000.0 if extra else latency_ms)


def _camera_source(value: Any) -> Union[int, str]:
    if isinstance(value, int):
        return value
    value = str(value)
    return int(value) if value.isdecimal() else value


def run_live(config: Mapping[str, Any], output_stream: TextIO) -> None:
    """Run the latest-frame loop until Ctrl-C or q in the optional preview window."""

    definitions = parse_brick_definitions(config)
    validate_files(config, definitions)
    camera_config = _required(config, "camera", "root")
    runtime_config = _required(config, "runtime", "root")
    backend = camera_config.get("backend", "opencv")
    if backend not in ("opencv", "dexmate"):
        raise ValueError(f"Unknown camera backend: {backend!r}")
    if backend == "dexmate":
        validate_dexmate_camera(camera_config)
    camera_matrix = np.asarray(_required(camera_config, "K", "camera"), dtype=np.float32)
    if camera_matrix.shape != (3, 3):
        raise ValueError("camera.K must be a 3x3 intrinsic matrix")

    detector = YoloBrickDetector(config["detector"], definitions)
    estimator = LiveBrickPoseEstimator(definitions, config["tracker"])
    target_hz = float(_required(runtime_config, "target_hz", "runtime"))
    if target_hz <= 0.0:
        raise ValueError("runtime.target_hz must be positive")
    period_s = 1.0 / target_hz
    timeout_s = float(_required(camera_config, "read_timeout_s", "camera"))
    preview_enabled = bool(_required(runtime_config, "preview", "runtime"))
    camera: Optional[Union[LatestFrameCamera, DexmateLatestFrameCamera]] = None
    show_grid = True
    show_axes = True
    window_name = "Live brick poses"
    sequence = -1
    expected_resolution = (
        int(_required(camera_config, "height", "camera")),
        int(_required(camera_config, "width", "camera")),
    )
    frame_id = 0
    next_tick = time.monotonic()
    try:
        if preview_enabled:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, expected_resolution[1], expected_resolution[0])
        if backend == "dexmate":
            camera = DexmateLatestFrameCamera(camera_config)
        else:
            camera = LatestFrameCamera(
                source=_camera_source(_required(camera_config, "source", "camera")),
                width=int(_required(camera_config, "width", "camera")),
                height=int(_required(camera_config, "height", "camera")),
                requested_fps=float(_required(camera_config, "requested_fps", "camera")),
            )
        while True:
            sleep_s = next_tick - time.monotonic()
            if sleep_s > 0.0:
                time.sleep(sleep_s)
            sequence, capture_timestamp_s, bgr = camera.read_latest(sequence, timeout_s)
            if bgr.shape[:2] != expected_resolution:
                raise RuntimeError(
                    f"Camera returned {bgr.shape[:2]} pixels but calibration is for "
                    f"{expected_resolution}; refusing to publish a metrically invalid pose"
                )
            frame_started = time.perf_counter()
            detector_started = time.perf_counter()
            detections = detector.detect(bgr)
            detector_latency_ms = (time.perf_counter() - detector_started) * 1000.0
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            poses, mode, megapose_latency_ms = estimator.estimate(rgb, camera_matrix, detections)
            total_latency_ms = (time.perf_counter() - frame_started) * 1000.0
            result = {
                # Version 1 is the initial stable JSONL contract. Consumers should reject unknown
                # versions; the integer is fixed until a breaking field/coordinate change occurs.
                "schema_version": 1,
                "frame_id": frame_id,
                "capture_timestamp_s": capture_timestamp_s,
                "publish_timestamp_s": time.time(),
                "pose_frame": "camera_optical",
                "mode": mode,
                "target_hz": target_hz,
                "deadline_missed": total_latency_ms > period_s * 1000.0,
                "detector_latency_ms": detector_latency_ms,
                "megapose_latency_ms": megapose_latency_ms,
                "total_latency_ms": total_latency_ms,
                "poses": poses,
            }
            if backend == "dexmate":
                result.update(
                    camera_backend=backend,
                    camera_frame_id=camera_config["frame_id"],
                    camera_sensor=camera_config.get("sensor", "head_camera"),
                    camera_image_key=camera_config.get("image_key", "left_rgb"),
                    capture_timestamp_source="host_receive_time",
                    sensor_capture_timestamp_s=None,
                    sensor_age_known=False,
                )
            output_stream.write(json.dumps(result, separators=(",", ":")) + "\n")
            output_stream.flush()

            if preview_enabled:
                cv2.imshow(window_name, draw_pose_preview(
                    bgr, poses, camera_matrix, mode, total_latency_ms,
                    grid=show_grid, axes=show_axes,
                ))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("g"):
                    show_grid = not show_grid
                if key == ord("a"):
                    show_axes = not show_axes
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break

            frame_id += 1
            next_tick += period_s
            if next_tick < time.monotonic():
                next_tick = time.monotonic()
    finally:
        if camera is not None:
            camera.close()
        estimator.close()
        if preview_enabled:
            cv2.destroyAllWindows()
