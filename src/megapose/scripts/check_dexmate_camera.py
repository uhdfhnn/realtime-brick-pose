"""Check calibrated Vega RGB reception without loading YOLO or MegaPose models."""
import argparse
import json
from pathlib import Path

import yaml

from megapose.dexmate_camera import DexmateLatestFrameCamera


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=30)
    args = parser.parse_args()
    if args.frames < 1:
        parser.error('--frames must be positive')
    config = yaml.safe_load(args.config.read_text())['camera']
    if config.get('backend') != 'dexmate':
        parser.error('camera.backend must be dexmate')
    camera = DexmateLatestFrameCamera(config)
    sequence = -1
    try:
        for _ in range(args.frames):
            sequence, received, bgr = camera.read_latest(sequence, config['read_timeout_s'])
            print(json.dumps(dict(sequence=sequence, received_timestamp_s=received,
                                  shape=list(bgr.shape), frame_id=config['frame_id'],
                                  timestamp_source=camera.timestamp_source)), flush=True)
    finally:
        camera.close()


if __name__ == '__main__':
    main()
