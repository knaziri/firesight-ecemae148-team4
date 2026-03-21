import cv2
import numpy as np


def pixel_to_world(px, py, H):
    pt    = np.array([[[float(px), float(py)]]], dtype=np.float32)
    world = cv2.perspectiveTransform(pt, H)
    return float(world[0][0][0]), float(world[0][0][1])


def get_marker_center(corners_array):
    c  = corners_array[0]
    cx = float(np.mean(c[:, 0]))
    cy = float(np.mean(c[:, 1]))
    return cx, cy


def get_marker_heading(corners_array):
    c  = corners_array[0]
    dx = c[1][0] - c[0][0]
    dy = c[1][1] - c[0][1]
    return float(np.degrees(np.arctan2(dy, dx)))


def build_arena_mask(corners, ids, frame_shape):
    """
    Returns a binary mask (same size as frame) where:
        255 = inside the arena boundary (between the 4 corner markers)
        0   = outside
    Returns None if not all 4 corner markers are visible.
    """
    from overhead_pkg.config import CORNER_IDS

    if ids is None:
        return None

    id_list = ids.flatten().tolist()
    if not all(cid in id_list for cid in CORNER_IDS):
        return None

    # Get the center of each corner marker in pixel space
    corner_pixels = []
    for cid in CORNER_IDS:
        idx = id_list.index(cid)
        cx, cy = get_marker_center(corners[idx])
        corner_pixels.append([int(cx), int(cy)])

    # Build a filled polygon mask from the 4 corner centers
    mask    = np.zeros(frame_shape[:2], dtype=np.uint8)
    polygon = np.array(corner_pixels, dtype=np.int32)
    cv2.fillConvexPoly(mask, polygon, 255)

    return mask
