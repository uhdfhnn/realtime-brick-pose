"""Color-based bootstrap detector for one blue 2x4 and one red 2x2 DUPLO."""
import cv2
import numpy as np


class ColorBrickDetector:
    def __init__(self, config):
        self.config = config
        for item in config['colors']:
            for low, high in item['hsv_ranges']:
                if len(low) != 3 or len(high) != 3 or any(a > b for a, b in zip(low, high)):
                    raise ValueError('Invalid HSV range')

    def detect(self, bgr):
        height, width = bgr.shape[:2]
        x1, y1, x2, y2 = self.config['roi_xyxy']
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError('Color detector ROI is outside the current image')
        hsv = cv2.cvtColor(bgr[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
        result = []
        for item in self.config['colors']:
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for low, high in item['hsv_ranges']:
                mask |= cv2.inRange(hsv, np.array(low, dtype=np.uint8), np.array(high, dtype=np.uint8))
            # Close tiny gaps between studs; no opening that would erase small red bricks.
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            candidates = []
            for i in range(1, count):
                x, y, w, h, area = stats[i]
                fill = area / float(w*h)
                if (self.config['min_area_px'] <= area <= self.config['max_area_px']
                        and fill >= self.config['min_fill_ratio'] and w >= 4 and h >= 4):
                    candidates.append((area, x, y, w, h, fill))
            if candidates:
                _, x, y, w, h, fill = max(candidates)
                pad = self.config.get('padding_px', 2)
                bbox = [max(x1, x+x1-pad), max(y1, y+y1-pad),
                        min(x2, x+x1+w+pad), min(y2, y+y1+h+pad)]
                result.append(dict(label=item['label'], bbox_xyxy=np.array(bbox, dtype=np.float32),
                                   confidence=float(fill)))
        return result
