"""CPU checks for moving color boxes, browser endpoints, and stream orchestration."""
import contextlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import urllib.request

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from megapose.color_bricks import ColorBrickDetector
from megapose.browser_preview import BrowserPreview
from megapose.scripts import stream_duplo


class ColorTest(unittest.TestCase):
    def setUp(self):
        self.config = yaml.safe_load((ROOT/'configs/live_duplo.colors.yaml').read_text())
        self.detector = ColorBrickDetector(self.config['color_detector'])

    def test_moves_and_disappears(self):
        a = np.zeros((1200,1920,3), np.uint8)
        a[666:700,925:974] = [255,0,0]
        a[683:713,1064:1097] = [0,0,255]
        boxes = {x['label']:x for x in self.detector.detect(a)}
        self.assertEqual(set(boxes), {'duplo_2x4','duplo_2x2'})
        b = np.zeros_like(a); b[730:764,970:1019] = [255,0,0]
        moved = self.detector.detect(b)
        self.assertEqual(len(moved),1)
        self.assertGreater(moved[0]['bbox_xyxy'][0],boxes['duplo_2x4']['bbox_xyxy'][0])
        self.assertEqual(self.detector.detect(np.zeros_like(a)), [])

    def test_outside_roi_and_tiny_noise_ignored(self):
        a=np.zeros((1200,1920,3),np.uint8)
        a[100:200,100:200]=[255,0,0]
        a[666:668,925:927]=[0,0,255]
        self.assertEqual(self.detector.detect(a),[])

    def test_invalid_resolution_fails(self):
        with self.assertRaises(ValueError):
            self.detector.detect(np.zeros((100,100,3),np.uint8))


class BrowserTest(unittest.TestCase):
    def test_status_and_jpeg(self):
        server=BrowserPreview(0)
        base=f'http://127.0.0.1:{server.port}'
        try:
            with urllib.request.urlopen(base+'/status') as r:
                self.assertIsNone(json.load(r)['image_age_s'])
            image=np.zeros((20,30,3),np.uint8);image[:]=[0,0,255]
            server.publish(image,'test frame')
            with urllib.request.urlopen(base+'/frame.jpg') as r:
                decoded=cv2.imdecode(np.frombuffer(r.read(),np.uint8),cv2.IMREAD_COLOR)
            self.assertEqual(decoded.shape,image.shape)
            self.assertGreater(decoded[5,5,2],240)
            with urllib.request.urlopen(base+'/status') as r:
                s=json.load(r)
            self.assertEqual(s['message'],'test frame')
            self.assertGreaterEqual(s['image_age_s'],0)
            with urllib.request.urlopen(base) as r:
                self.assertIn(b'DUPLO live inspection',r.read())
        finally:
            server.close()


class RunnerTest(unittest.TestCase):
    def run_mode(self,camera_only):
        events=[]
        image=np.zeros((1200,1920,3),np.uint8);image[666:700,925:974]=[255,0,0]
        class Camera:
            def __init__(self,config):self.reads=0
            def read_latest(self,*args):
                self.reads+=1
                if self.reads>1:raise KeyboardInterrupt
                return 0,123.,image.copy()
            def close(self):events.append('camera closed')
        class Viewer:
            port=8090
            def __init__(self,*args):pass
            def publish(self,image,message):events.append(message)
            def status(self,message):pass
            def close(self):events.append('viewer closed')
        class Estimator:
            def __init__(self,*args):pass
            def estimate(self,rgb,K,detections):
                self_test.assertEqual(len(detections),1)
                self_test.assertEqual(detections[0].label,'duplo_2x4')
                self_test.assertEqual(rgb[680,940].tolist(),[0,0,255])
                events.append('estimated')
                return [],'no_detection',1.
            def close(self):events.append('estimator closed')
        self_test=self
        fake=types.ModuleType('megapose.live_bricks')
        fake.Detection=lambda label,bbox,conf:types.SimpleNamespace(label=label)
        fake.LiveBrickPoseEstimator=Estimator
        fake.parse_brick_definitions=lambda config:[]
        torch=types.ModuleType('torch');torch.inference_mode=contextlib.nullcontext
        with tempfile.TemporaryDirectory() as d:
            output=Path(d)/'poses.jsonl'
            argv=['stream','--config',str(ROOT/'configs/live_duplo.colors.yaml'),
                  '--output-jsonl',str(output)]
            if camera_only:argv+=['--camera-only']
            with patch.object(sys,'argv',argv), patch.object(stream_duplo,'DexmateLatestFrameCamera',Camera), \
                    patch.object(stream_duplo,'BrowserPreview',Viewer), \
                    patch.dict(sys.modules,{'megapose.live_bricks':fake,'torch':torch}):
                stream_duplo.main()
            data=json.loads(output.read_text())
            self.assertEqual(data['candidates'][0]['label'],'duplo_2x4')
            self.assertEqual(data['capture_timestamp_source'],'host_receive_time')
        self.assertIn('camera closed',events);self.assertIn('viewer closed',events)
        self.assertEqual('estimated' in events,not camera_only)
        self.assertEqual('estimator closed' in events,not camera_only)

    def test_camera_only(self):self.run_mode(True)
    def test_pose_mode(self):self.run_mode(False)


if __name__=='__main__':unittest.main()
