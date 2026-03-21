import cv2
import numpy as np
from overhead_pkg.utils import get_marker_center
from overhead_pkg.config import CORNER_IDS, REAL_WORLD_CORNERS


def try_compute_homography(corners, ids):
    """
    Attempts to compute homography from current frame.
    Returns (H, H_inv) if all 4 corners visible, else (None, None).
    """
    if ids is None:
        return None, None

    id_list = ids.flatten().tolist()
    if not all(cid in id_list for cid in CORNER_IDS):
        return None, None

    pixel_pts = []
    for cid in CORNER_IDS:
        idx = id_list.index(cid)
        cx, cy = get_marker_center(corners[idx])
        pixel_pts.append([cx, cy])

    pixel_pts = np.array(pixel_pts, dtype=np.float32)
    H, _      = cv2.findHomography(pixel_pts, REAL_WORLD_CORNERS)
    H_inv     = np.linalg.inv(H)
    return H, H_inv
