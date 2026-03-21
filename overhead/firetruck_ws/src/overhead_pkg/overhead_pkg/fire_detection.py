import cv2
import numpy as np
from overhead_pkg.utils import pixel_to_world
from overhead_pkg.config import FIRE_HSV_LOWER, FIRE_HSV_UPPER, FIRE_MIN_AREA


def detect_fires(frame, H):
    """
    Returns list of dicts:
        px, py  : pixel centroid
        wx, wy  : world centroid (cm)
        bbox    : (x, y, w, h) pixel bounding box
    """
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, FIRE_HSV_LOWER, FIRE_HSV_UPPER)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    fires = []
    for cnt in contours:
        if cv2.contourArea(cnt) < FIRE_MIN_AREA:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        M           = cv2.moments(cnt)
        if M['m00'] == 0:
            continue
        px = int(M['m10'] / M['m00'])
        py = int(M['m01'] / M['m00'])
        wx, wy = pixel_to_world(px, py, H)
        fires.append({'px': px, 'py': py, 'wx': wx, 'wy': wy, 'bbox': (x, y, w, h)})

    return fires


def draw_fires(display, fires):
    for i, fire in enumerate(fires):
        x, y, w, h = fire['bbox']
        cv2.rectangle(display, (x, y), (x + w, y + h), (0, 69, 255), 2)
        cv2.circle(display, (fire['px'], fire['py']), 4, (0, 69, 255), -1)
        cv2.putText(display,
            f"Fire {i+1} ({fire['wx']:.1f},{fire['wy']:.1f})cm",
            (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 69, 255), 2)
