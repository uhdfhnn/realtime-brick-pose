"""Focused tests for live brick config, detection conversion, and symmetry continuity."""

# Standard Library
from pathlib import Path

# Third Party
import numpy as np

# MegaPose
from megapose.live_bricks import (
    Detection,
    bbox_iou,
    canonicalize_pose,
    detections_to_tensor,
    parse_brick_definitions,
)


def test_bbox_iou() -> None:
    left = np.array([0.0, 0.0, 10.0, 10.0])
    right = np.array([5.0, 0.0, 15.0, 10.0])
    assert np.isclose(bbox_iou(left, right), 1.0 / 3.0)


def test_detection_conversion() -> None:
    if not __import__("torch").cuda.is_available():
        __import__("pytest").skip("MegaPose detections are intentionally placed on CUDA")
    detections = [Detection("brick_2x4", np.array([1, 2, 30, 40]), 0.9)]
    tensor = detections_to_tensor(detections)
    assert tensor.infos.iloc[0]["label"] == "brick_2x4"
    assert tensor.bboxes.cpu().numpy().tolist() == [[1, 2, 30, 40]]


def test_canonicalization_uses_previous_symmetric_representative() -> None:
    identity = np.eye(4)
    half_turn = np.diag([-1.0, -1.0, 1.0, 1.0])
    raw_pose = half_turn.copy()
    canonical, symmetry_index = canonicalize_pose(
        raw_pose, np.stack([identity, half_turn]), previous_pose=identity
    )
    assert symmetry_index == 1
    assert np.allclose(canonical, identity)


def test_parse_three_bricks() -> None:
    config = {
        "objects": [
            {
                "label": name,
                "yolo_class_name": name,
                "mesh_path": str(Path("meshes") / f"{name}.ply"),
                "dimensions_mm": dimensions,
                "object_frame": {
                    "origin": "center",
                    "axes": {"x": "length", "y": "width", "z": "up"},
                },
                "symmetry": {"discrete_transforms": []},
            }
            for name, dimensions in (
                ("brick_2x4", [31.8, 15.8, 9.6]),
                ("brick_2x2", [15.8, 15.8, 9.6]),
                ("brick_1x2", [15.8, 7.8, 9.6]),
            )
        ]
    }
    definitions = parse_brick_definitions(config)
    assert [definition.label for definition in definitions] == [
        "brick_2x4",
        "brick_2x2",
        "brick_1x2",
    ]
