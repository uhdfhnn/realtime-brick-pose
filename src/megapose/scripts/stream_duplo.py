"""Stream blue-2x4/red-2x2 color proposals and optional MegaPose estimates to a browser."""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import yaml

from megapose.browser_preview import BrowserPreview
from megapose.color_bricks import ColorBrickDetector
from megapose.dexmate_camera import DexmateLatestFrameCamera
from megapose.brick_preview import draw_pose_preview


def draw_detections(bgr, detections, roi):
    image = bgr.copy()
    cv2.rectangle(image, tuple(roi[:2]), tuple(roi[2:]), (160, 160, 160), 1)
    for item in detections:
        x1, y1, x2, y2 = item['bbox_xyxy'].astype(int)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(image, item['label']+' (color)', (x1, max(20, y1-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1)
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/live_duplo.colors.yaml'))
    parser.add_argument('--port', type=int, default=8090)
    parser.add_argument('--camera-only', action='store_true', help='Stream color boxes without pose models')
    parser.add_argument('--output-jsonl', type=Path, default=Path('local_data/duplo_stream.jsonl'))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    for obj in config['objects']:
        obj['mesh_path'] = str((args.config.resolve().parent / obj['mesh_path']).resolve())
    detector = ColorBrickDetector(config['color_detector'])
    target_hz = float(config['runtime']['target_hz'])
    if not np.isfinite(target_hz) or target_hz <= 0:
        raise ValueError('target_hz must be finite and positive')
    viewer = BrowserPreview(args.port)
    print(f'Open http://127.0.0.1:{viewer.port} on the Linux desktop', flush=True)
    camera = estimator = None
    try:
        if not args.camera_only:
            viewer.status('Loading MegaPose models; camera will start after loading')
            from megapose.live_bricks import Detection, LiveBrickPoseEstimator, parse_brick_definitions
            definitions = parse_brick_definitions(config)
            for definition in definitions:
                if not definition.mesh_path.is_file():
                    raise FileNotFoundError(definition.mesh_path)
            estimator = LiveBrickPoseEstimator(definitions, config['tracker'])
        viewer.status('Connecting to head camera')
        camera = DexmateLatestFrameCamera(config['camera'])
        K = np.asarray(config['camera']['K'], dtype=np.float32)
        sequence = -1
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.output_jsonl.open('a') as stream:
            while True:
                start = time.monotonic()
                sequence, received, bgr = camera.read_latest(sequence, config['camera']['read_timeout_s'])
                proposals = detector.detect(bgr)
                detection_image = draw_detections(bgr, proposals, config['color_detector']['roi_xyxy'])
                message = f'{len(proposals)} color candidates | ' + (
                    'camera-only' if args.camera_only else 'estimating next frame...')
                if args.camera_only or sequence == 0:
                    viewer.publish(detection_image, message)
                else:
                    # Keep the last completed frame and its matching poses visible.
                    viewer.status(message)
                poses, mode, latency = [], 'camera_only', 0.
                if estimator is not None:
                    import torch
                    detections = [Detection(x['label'], x['bbox_xyxy'], x['confidence']) for x in proposals]
                    with torch.inference_mode():
                        poses, mode, latency = estimator.estimate(
                            cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), K, detections)
                    for pose in poses:
                        pose.update(detection_source='hsv_color',
                                    detection_confidence_kind='mask_fill_ratio_not_probability')
                    image = draw_pose_preview(bgr, poses, K, mode, latency)
                    roi = config['color_detector']['roi_xyxy']
                    cv2.rectangle(image, tuple(roi[:2]), tuple(roi[2:]), (160,160,160), 1)
                    # Replace desktop keyboard hints; this viewer is controlled through Ctrl+C.
                    cv2.rectangle(image, (0, 24), (bgr.shape[1]-1, 45), (0, 0, 0), -1)
                    cv2.putText(image, 'Color proposals: blue=2x4 red=2x2 | Ctrl+C in terminal stops',
                                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, .45, (255,255,255), 1)
                    viewer.publish(image, f'{mode} | {len(poses)} poses | inference {latency:.0f} ms')
                result = dict(schema_version=1, frame_id=sequence, capture_timestamp_s=received,
                              capture_timestamp_source='host_receive_time', sensor_age_known=False,
                              publish_timestamp_s=time.time(), pose_frame='camera_optical',
                              camera_frame_id=config['camera']['frame_id'], detection_source='hsv_color',
                              mode=mode, poses=poses,
                              candidates=[dict(label=x['label'], bbox_xyxy_px=x['bbox_xyxy'].tolist(),
                                               mask_fill_ratio=x['confidence']) for x in proposals])
                stream.write(json.dumps(result)+'\n')
                stream.flush()
                time.sleep(max(0., 1./target_hz - (time.monotonic()-start)))
    except KeyboardInterrupt:
        print('Stopped', flush=True)
    finally:
        try:
            if camera is not None:
                camera.close()
        finally:
            try:
                if estimator is not None:
                    estimator.close()
            finally:
                viewer.close()


if __name__ == '__main__':
    main()
