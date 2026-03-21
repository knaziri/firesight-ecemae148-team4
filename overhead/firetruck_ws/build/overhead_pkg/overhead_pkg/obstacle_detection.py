import cv2
import numpy as np
from scipy.ndimage import binary_dilation
from overhead_pkg.config import (
    BLUE_HSV_LOWER, BLUE_HSV_UPPER,
    OBSTACLE_MIN_AREA, OBSTACLE_OVERLAY_ALPHA,
    OBSTACLE_PADDING_CELLS, OCCUPANCY_GRID_RESOLUTION,
    ARENA_WIDTH_CM, ARENA_HEIGHT_CM
)
from overhead_pkg.utils import pixel_to_world


def build_obstacle_mask(frame, arena_mask):
    """
    Returns a binary mask where:
        255 = blue obstacle pixel within arena bounds
        0   = everything else
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Detect blue pixels only
    blue_mask = cv2.inRange(hsv, BLUE_HSV_LOWER, BLUE_HSV_UPPER)

    # Restrict to arena bounds
    obstacle_mask = cv2.bitwise_and(blue_mask, arena_mask)

    # Morphological cleanup
    kernel        = np.ones((5, 5), np.uint8)
    obstacle_mask = cv2.morphologyEx(obstacle_mask, cv2.MORPH_OPEN,  kernel)
    obstacle_mask = cv2.morphologyEx(obstacle_mask, cv2.MORPH_CLOSE, kernel)

    return obstacle_mask


def get_obstacle_contours(obstacle_mask):
    contours, _ = cv2.findContours(
        obstacle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) > OBSTACLE_MIN_AREA]


def build_occupancy_grid(obstacle_mask, H):
    grid_w = int(ARENA_WIDTH_CM  / OCCUPANCY_GRID_RESOLUTION)
    grid_h = int(ARENA_HEIGHT_CM / OCCUPANCY_GRID_RESOLUTION)
    grid   = np.zeros((grid_h, grid_w), dtype=np.uint8)

    obstacle_pixels = np.argwhere(obstacle_mask > 0)
    for py, px in obstacle_pixels[::4]:
        try:
            wx, wy = pixel_to_world(px, py, H)
            gx = int(wx / OCCUPANCY_GRID_RESOLUTION)
            gy = int(wy / OCCUPANCY_GRID_RESOLUTION)
            if 0 <= gx < grid_w and 0 <= gy < grid_h:
                grid[gy, gx] = 1
        except Exception:
            continue

    if OBSTACLE_PADDING_CELLS > 0:
        struct = np.ones((
            OBSTACLE_PADDING_CELLS * 2 + 1,
            OBSTACLE_PADDING_CELLS * 2 + 1
        ))
        grid = binary_dilation(grid, structure=struct).astype(np.uint8)

    return grid


def draw_obstacles(display, obstacle_contours):
    overlay = display.copy()
    cv2.drawContours(overlay, obstacle_contours, -1, (0, 0, 180), thickness=cv2.FILLED)
    cv2.addWeighted(overlay, OBSTACLE_OVERLAY_ALPHA,
                    display, 1 - OBSTACLE_OVERLAY_ALPHA, 0, display)
    cv2.drawContours(display, obstacle_contours, -1, (0, 0, 255), 2)

    for i, cnt in enumerate(obstacle_contours):
        M = cv2.moments(cnt)
        if M['m00'] == 0:
            continue
        cx   = int(M['m10'] / M['m00'])
        cy   = int(M['m01'] / M['m00'])
        area = cv2.contourArea(cnt)
        cv2.putText(display, f"Obs {i+1} ({area:.0f}px)",
            (cx - 30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)


def draw_arena_boundary(display, corners, ids):
    from overhead_pkg.config import CORNER_IDS
    from overhead_pkg.utils  import get_marker_center

    if ids is None:
        return

    id_list = ids.flatten().tolist()
    if not all(cid in id_list for cid in CORNER_IDS):
        return

    pts = []
    for cid in CORNER_IDS:
        idx = id_list.index(cid)
        cx, cy = get_marker_center(corners[idx])
        pts.append([int(cx), int(cy)])

    polygon = np.array(pts, dtype=np.int32)
    cv2.polylines(display, [polygon], isClosed=True, color=(0, 255, 255), thickness=2)


def draw_occupancy_grid(display, grid, H_inv):
    grid_h, grid_w = grid.shape
    for gy in range(grid_h):
        for gx in range(grid_w):
            if grid[gy, gx] == 1:
                wx  = gx * OCCUPANCY_GRID_RESOLUTION + OCCUPANCY_GRID_RESOLUTION / 2
                wy  = gy * OCCUPANCY_GRID_RESOLUTION + OCCUPANCY_GRID_RESOLUTION / 2
                pt  = np.array([[[wx, wy]]], dtype=np.float32)
                px  = cv2.perspectiveTransform(pt, H_inv)
                ppx = int(px[0][0][0])
                ppy = int(px[0][0][1])
                cv2.circle(display, (ppx, ppy), 2, (0, 0, 255), -1)
