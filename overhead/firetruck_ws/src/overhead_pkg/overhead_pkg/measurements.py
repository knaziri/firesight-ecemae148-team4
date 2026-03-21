import cv2
import numpy as np
from overhead_pkg.pathfinding import (
    astar, smooth_path, world_to_grid,
    grid_to_world, path_length_cm
)


def compute_paths(robot, fires, grid, H_inv):
    """
    For each fire, runs A* from robot to fire on the occupancy grid.
    Returns list of dicts per fire:
        path_grid    : smoothed list of (row, col) grid cells
        path_pixels  : list of (px, py) pixel coordinates for drawing
        length_cm    : total path length in cm
    """
    if robot is None or grid is None:
        return []

    results = []

    for fire in fires:
        start_grid = world_to_grid(robot['wx'], robot['wy'])
        goal_grid  = world_to_grid(fire['wx'],  fire['wy'])

        raw_path = astar(grid, start_grid, goal_grid)

        if raw_path is None:
            results.append({
                'path_grid':   None,
                'path_pixels': None,
                'length_cm':   None
            })
            continue

        smoothed   = smooth_path(raw_path, grid)
        length_cm  = path_length_cm(smoothed)

        # Convert grid cells to pixel coordinates via inverse homography
        path_pixels = []
        for row, col in smoothed:
            wx, wy = grid_to_world(row, col)
            pt     = np.array([[[wx, wy]]], dtype=np.float32)
            px     = cv2.perspectiveTransform(pt, H_inv)
            path_pixels.append((int(px[0][0][0]), int(px[0][0][1])))

        results.append({
            'path_grid':   smoothed,
            'path_pixels': path_pixels,
            'length_cm':   length_cm
        })

    return results


def draw_paths(display, paths, fires):
    """
    Draws each path as a polyline with waypoint dots and a length label.
    """
    colors = [
        (255, 200,   0),   # fire 1 - yellow
        (255,   0, 200),   # fire 2 - magenta
        (  0, 200, 255),   # fire 3 - cyan
        (200, 255,   0),   # fire 4 - lime
    ]

    for i, (path_data, fire) in enumerate(zip(paths, fires)):
        color = colors[i % len(colors)]

        if path_data['path_pixels'] is None:
            # No path found - draw red X at fire location
            fx, fy = fire['px'], fire['py']
            cv2.line(display, (fx-10, fy-10), (fx+10, fy+10), (0, 0, 255), 2)
            cv2.line(display, (fx+10, fy-10), (fx-10, fy+10), (0, 0, 255), 2)
            cv2.putText(display, f"Fire {i+1}: NO PATH",
                (fx + 12, fy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            continue

        pts = path_data['path_pixels']

        # Draw path polyline
        for j in range(len(pts) - 1):
            cv2.line(display, pts[j], pts[j + 1], color, 2)

        # Draw waypoint dots
        for pt in pts[1:-1]:
            cv2.circle(display, pt, 4, color, -1)

        # Length label at midpoint of path
        if len(pts) >= 2:
            mid_idx = len(pts) // 2
            mx, my  = pts[mid_idx]
            cv2.putText(display,
                f"Fire {i+1}: {path_data['length_cm']:.1f}cm",
                (mx + 6, my - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
