import heapq
import numpy as np
import cv2
from overhead_pkg.config import OCCUPANCY_GRID_RESOLUTION


def heuristic(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def astar(grid, start, goal):
    """
    A* on occupancy grid.
    start, goal: (row, col) grid coordinates.
    Returns list of (row, col) cells from start to goal, or None if no path.
    """
    rows, cols = grid.shape

    # Clamp start and goal to grid bounds
    start = (
        int(np.clip(start[0], 0, rows - 1)),
        int(np.clip(start[1], 0, cols - 1))
    )
    goal = (
        int(np.clip(goal[0], 0, rows - 1)),
        int(np.clip(goal[1], 0, cols - 1))
    )

    # If start or goal is inside an obstacle, find nearest free cell
    start = nearest_free(grid, start)
    goal  = nearest_free(grid, goal)

    if start is None or goal is None:
        return None

    open_set = []
    heapq.heappush(open_set, (0, start))

    came_from = {}
    g_score   = {start: 0}
    f_score   = {start: heuristic(start, goal)}

    # 8-directional movement
    neighbors = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1,-1), (-1, 1), (1,-1), (1, 1)
    ]

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            return reconstruct_path(came_from, current)

        for dr, dc in neighbors:
            neighbor = (current[0] + dr, current[1] + dc)
            nr, nc   = neighbor

            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if grid[nr, nc] == 1:
                continue

            move_cost       = 1.4 if dr != 0 and dc != 0 else 1.0
            tentative_g     = g_score[current] + move_cost

            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor]   = tentative_g
                f_score[neighbor]   = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None  # no path found


def reconstruct_path(came_from, current):
    path = []
    while current in came_from:
        path.append(current)
        current = came_from[current]
    path.append(current)
    return path[::-1]


def nearest_free(grid, cell):
    """
    If cell is occupied, spiral outward to find nearest free cell.
    """
    rows, cols = grid.shape
    if grid[cell[0], cell[1]] == 0:
        return cell

    for radius in range(1, max(rows, cols)):
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = cell[0] + dr, cell[1] + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if grid[nr, nc] == 0:
                        return (nr, nc)
    return None


def smooth_path(path, grid):
    """
    Prunes redundant waypoints using line-of-sight checks.
    Returns a shorter list of waypoints.
    """
    if len(path) <= 2:
        return path

    smoothed = [path[0]]
    i = 0

    while i < len(path) - 1:
        for j in range(len(path) - 1, i, -1):
            if line_of_sight(path[i], path[j], grid):
                smoothed.append(path[j])
                i = j
                break
        else:
            i += 1

    return smoothed


def line_of_sight(p1, p2, grid):
    """
    Bresenham's line algorithm - checks if straight line between
    two grid cells is free of obstacles.
    """
    r1, c1 = p1
    r2, c2 = p2
    rows, cols = grid.shape

    dr = abs(r2 - r1)
    dc = abs(c2 - c1)
    sr = 1 if r1 < r2 else -1
    sc = 1 if c1 < c2 else -1
    err = dr - dc

    while True:
        if not (0 <= r1 < rows and 0 <= c1 < cols):
            return False
        if grid[r1, c1] == 1:
            return False
        if r1 == r2 and c1 == c2:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r1  += sr
        if e2 < dr:
            err += dr
            c1  += sc

    return True


def world_to_grid(wx, wy):
    col = int(wx / OCCUPANCY_GRID_RESOLUTION)
    row = int(wy / OCCUPANCY_GRID_RESOLUTION)
    return (row, col)


def grid_to_world(row, col):
    wx = col * OCCUPANCY_GRID_RESOLUTION + OCCUPANCY_GRID_RESOLUTION / 2
    wy = row * OCCUPANCY_GRID_RESOLUTION + OCCUPANCY_GRID_RESOLUTION / 2
    return (wx, wy)


def path_length_cm(path):
    """
    Computes total path length in cm from a list of grid cells.
    """
    total = 0.0
    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i + 1]
        total += np.sqrt(
            ((r2 - r1) * OCCUPANCY_GRID_RESOLUTION)**2 +
            ((c2 - c1) * OCCUPANCY_GRID_RESOLUTION)**2
        )
    return total
