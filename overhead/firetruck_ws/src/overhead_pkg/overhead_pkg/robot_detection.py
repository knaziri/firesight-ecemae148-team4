import cv2
import numpy as np
from overhead_pkg.utils import get_marker_center, get_marker_heading, pixel_to_world
from overhead_pkg.config import ROBOT_ID


def detect_robot(corners, ids, H):
    """
    Returns dict with robot state or None if not detected:
        px, py          : pixel position
        wx, wy          : world position (cm)
        heading         : degrees
    """
    if ids is None:
        return None

    id_list = ids.flatten().tolist()
    if ROBOT_ID not in id_list:
        return None

    idx           = id_list.index(ROBOT_ID)
    rpx, rpy      = get_marker_center(corners[idx])
    rpx, rpy      = int(rpx), int(rpy)
    rwx, rwy      = pixel_to_world(rpx, rpy, H)
    heading       = get_marker_heading(corners[idx])

    return {'px': rpx, 'py': rpy, 'wx': rwx, 'wy': rwy, 'heading': heading}


def draw_robot(display, robot):
    if robot is None:
        return
    heading_rad = np.radians(robot['heading'])
    arrow_tip   = (
        int(robot['px'] + 50 * np.cos(heading_rad)),
        int(robot['py'] + 50 * np.sin(heading_rad))
    )
    cv2.arrowedLine(display, (robot['px'], robot['py']),
                    arrow_tip, (0, 255, 0), 2, tipLength=0.3)
    cv2.putText(display,
        f"Robot ({robot['wx']:.1f},{robot['wy']:.1f})cm",
        (robot['px'] + 10, robot['py'] - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
