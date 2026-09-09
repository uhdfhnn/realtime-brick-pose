"""Camera-image overlay for metric brick poses; NumPy projection is CPU-only."""
from itertools import product

import numpy as np


def project_points(points_m, pose, K):
    """Return optical pinhole pixels and validity, rejecting near/behind-camera points."""
    points = np.asarray(points_m, dtype=float)
    T = np.asarray(pose, dtype=float)
    camera = points @ T[:3, :3].T + T[:3, 3]
    homogeneous = camera @ np.asarray(K, dtype=float).T
    valid = np.isfinite(homogeneous).all(axis=1) & (camera[:, 2] > 1e-4)
    pixels = np.full((len(points), 2), np.nan)
    pixels[valid] = homogeneous[valid, :2] / homogeneous[valid, 2, None]
    # Keep conversion to OpenCV int32 coordinates bounded for corrupt estimates.
    valid &= np.isfinite(pixels).all(axis=1) & (np.abs(pixels) < 1e7).all(axis=1)
    return pixels, valid


def draw_pose_preview(bgr, poses, K, mode, latency_ms, *, grid=True, axes=True):
    """Overlay a pixel grid, centered nominal cuboids, and object-frame XYZ axes.

    Cuboids assume the existing config's geometric-center object origin. They are
    nominal dimension envelopes, not CAD mesh renders or collision geometry.
    """
    import cv2

    canvas = bgr.copy()
    height, width = canvas.shape[:2]
    if grid:
        for x in range(0, width, 80):
            cv2.line(canvas, (x, 0), (x, height - 1), (65, 65, 65), 1)
        for y in range(0, height, 80):
            cv2.line(canvas, (0, y), (width - 1, y), (65, 65, 65), 1)

    def text(value, xy, color=(255, 255, 255), scale=.45):
        cv2.putText(canvas, value, xy, cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, value, xy, cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, 1, cv2.LINE_AA)

    def segment(uv, valid, a, b, color, thickness=2):
        if valid[a] and valid[b]:
            start = tuple(np.rint(uv[a]).astype(int))
            end = tuple(np.rint(uv[b]).astype(int))
            ok, start, end = cv2.clipLine((0, 0, width, height), start, end)
            if ok:
                cv2.line(canvas, start, end, color, thickness, cv2.LINE_AA)

    for index, pose in enumerate(poses):
        color = (0, 230, 0) if pose['valid'] else (0, 165, 255)
        bbox = np.asarray(pose['bbox_xyxy_px'], dtype=float)
        if np.isfinite(bbox).all():
            bbox = np.clip(bbox, [0, 0, 0, 0], [width-1, height-1, width-1, height-1])
            x1, y1, x2, y2 = np.rint(bbox).astype(int)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1)
        dimensions = np.asarray(pose['mesh_dimensions_mm'], dtype=float) / 1000
        T = np.asarray(pose['T_camera_object_m'], dtype=float)
        if axes:
            if pose.get('object_frame', {}).get('origin') == 'geometric center of the nominal outer bounding box':
                corners = np.array(list(product((-1, 1), repeat=3))) * dimensions / 2
                uv, valid = project_points(corners, T, K)
                for a in range(8):
                    for bit in (1, 2, 4):
                        b = a ^ bit
                        if a < b:
                            segment(uv, valid, a, b, color)
            axis_length = float(max(dimensions) * .8)
            endpoints = np.vstack([np.zeros(3), np.eye(3) * axis_length])
            uv, valid = project_points(endpoints, T, K)
            for i, (name, axis_color) in enumerate(
                    [('X', (0, 0, 255)), ('Y', (0, 255, 0)), ('Z', (255, 0, 0))], 1):
                segment(uv, valid, 0, i, axis_color, 2)
                if valid[i] and 0 <= uv[i, 0] < width and 0 <= uv[i, 1] < height:
                    text(name, tuple(np.rint(uv[i]).astype(int)), axis_color)
        xyz = np.asarray(pose['position_m']) * 1000
        status = 'SCORE PASS' if pose['valid'] else 'LOW SCORE'
        text(f"{pose['label']}  xyz(mm)=({xyz[0]:.1f}, {xyz[1]:.1f}, {xyz[2]:.1f})"
             f"  det={pose['detection_confidence']:.2f} pose={pose['pose_score']:.2f} {status}",
             (10, 62 + index * 22), color)

    text(f'{mode} | inference {latency_ms:.1f} ms | camera optical frame', (10, 20))
    text('g: pixel grid (80px) | a: pose axes/box | q/Esc: quit', (10, 40))
    if not poses:
        text('No brick detected', (10, 65), (0, 165, 255))
    return canvas
