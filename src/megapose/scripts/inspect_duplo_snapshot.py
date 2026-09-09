"""Estimate two DUPLO poses from a saved RGB image and explicit manual boxes."""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import yaml

from megapose.brick_preview import draw_pose_preview, project_points

ROOT = Path(__file__).resolve().parents[3]


def parse_boxes(values, width, height):
    boxes = []
    allowed = {'duplo_2x2', 'duplo_2x4'}
    for label, *coords in values:
        bbox = np.asarray(coords, dtype=float)
        if label not in allowed or any(x['label'] == label for x in boxes):
            raise ValueError('Use each of duplo_2x2 and duplo_2x4 at most once')
        x1, y1, x2, y2 = bbox
        if not np.isfinite(bbox).all() or not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f'Invalid box for {label}: {bbox}')
        boxes.append(dict(label=label, bbox=bbox))
    return boxes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--box', nargs=5, action='append', required=True,
                        metavar=('LABEL', 'XMIN', 'YMIN', 'XMAX', 'YMAX'))
    parser.add_argument('--output-dir', type=Path, default=Path('local_data/duplo_inspection'))
    parser.add_argument('--prepare-only', action='store_true', help='Save box preview without models')
    args = parser.parse_args()
    bgr = cv2.imread(str(args.image))
    if bgr is None:
        raise ValueError(f'Cannot read {args.image}')
    cam = yaml.safe_load(args.config.read_text())['camera']
    from megapose.dexmate_camera import validate_dexmate_camera
    K = validate_dexmate_camera(cam)
    height, width = bgr.shape[:2]
    if (height, width) != (cam['height'], cam['width']):
        raise ValueError('Image resolution does not match calibration')
    boxes = parse_boxes(args.box, width, height)
    metadata = json.loads((ROOT / 'assets/duplo/meshes.json').read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    annotated = bgr.copy()
    for item in boxes:
        x1, y1, x2, y2 = item['bbox'].astype(int)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 1)
        cv2.putText(annotated, item['label'], (x1, max(15, y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 255, 255), 1)
    cv2.imwrite(str(args.output_dir / 'detections.png'), annotated)
    if args.prepare_only:
        print('Saved', args.output_dir / 'detections.png')
        return

    import pandas as pd
    import torch
    import trimesh
    from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset
    from megapose.inference.types import ObservationTensor
    from megapose.lib3d.transform import Transform
    from megapose.utils.load_model import load_named_model
    from megapose.utils.tensor_collection import PandasTensorCollection

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA PyTorch is required')
    mesh_paths = {x['label']: ROOT / 'assets/duplo' / metadata[x['label']]['mesh'] for x in boxes}
    dataset = RigidObjectDataset([RigidObject(label=k, mesh_path=v, mesh_units='mm')
                                  for k, v in mesh_paths.items()])
    detections = PandasTensorCollection(
        infos=pd.DataFrame([dict(label=x['label'], batch_im_id=0, instance_id=i, score=1.)
                            for i, x in enumerate(boxes)]),
        bboxes=torch.tensor(np.stack([x['bbox'] for x in boxes]), dtype=torch.float32),
    ).cuda()
    observation = ObservationTensor.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), K=K).cuda()
    estimator = None
    try:
        estimator = load_named_model('megapose-1.0-RGB', dataset, n_workers=1, bsz_images=32).cuda()
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            output, _ = estimator.run_inference_pipeline(
                observation, detections=detections, n_refiner_iterations=5, n_pose_hypotheses=1)
        torch.cuda.synchronize()
        latency_ms = (time.perf_counter()-start) * 1000
        poses = []
        masks = []
        for index, row in output.infos.reset_index(drop=True).iterrows():
            label = str(row['label'])
            T = output.poses[index].detach().cpu().numpy()
            transform = Transform(T)
            score = float(row['pose_score'])
            box = next(x for x in boxes if x['label'] == label)
            poses.append(dict(label=label, bbox_xyxy_px=box['bbox'].tolist(),
                              detection_confidence=1., detection_source='manual',
                              pose_score=score, valid=bool(np.isfinite(score) and score >= .5),
                              T_camera_object_m=T.tolist(), position_m=T[:3, 3].tolist(),
                              quaternion_xyzw=transform.quaternion.coeffs().tolist(),
                              mesh_dimensions_mm=metadata[label]['dimensions_mm'],
                              object_frame=dict(origin='geometric center of the nominal outer bounding box',
                                                axes=metadata[label]['axes'])))
            mesh = trimesh.load(str(mesh_paths[label]), force='mesh', process=False)
            uv, valid = project_points(np.asarray(mesh.vertices)/1000, T, K)
            mask = np.zeros((height, width), dtype=np.uint8)
            for face in mesh.faces:
                if valid[face].all():
                    cv2.fillConvexPoly(mask, np.rint(uv[face]).astype(np.int32), 255)
            masks.append(mask)
        result = dict(pose_frame='camera_optical', camera_frame_id=cam['frame_id'],
                      image=str(args.image), K=K.tolist(), resolution=[height, width],
                      inference_latency_ms=latency_ms, detection_source='manual', poses=poses)
        (args.output_dir / 'poses.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        overlay = bgr.copy()
        for mask, color in zip(masks, [(0, 255, 0), (255, 0, 255)]):
            selected = mask > 0
            overlay[selected] = (.65 * overlay[selected] + .35 * np.array(color)).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, color, 1)
        cv2.imwrite(str(args.output_dir / 'mesh_overlay.png'), overlay)
        axes = draw_pose_preview(bgr, poses, K, 'snapshot', latency_ms, grid=True, axes=True)
        cv2.imwrite(str(args.output_dir / 'pose_overlay.png'), axes)
        print('Wrote poses.json, mesh_overlay.png, pose_overlay.png to', args.output_dir)
    finally:
        if estimator is not None:
            renderers = {id(m.renderer): m.renderer for m in
                         (estimator.coarse_model, estimator.refiner_model)}
            for renderer in renderers.values():
                renderer.stop()


if __name__ == '__main__':
    main()
