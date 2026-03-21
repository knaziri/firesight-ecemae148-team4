import numpy as np

# --- Arena dimensions in cm ---
ARENA_WIDTH_CM  = 300.0
ARENA_HEIGHT_CM = 180.0

# --- ArUco IDs ---
CORNER_IDS = [0, 1, 2, 3]   # TL, TR, BR, BL
ROBOT_ID   = 10

# Real-world positions of corner markers (cm): TL, TR, BR, BL
REAL_WORLD_CORNERS = np.array([
    [0.0,            0.0            ],
    [ARENA_WIDTH_CM, 0.0            ],
    [ARENA_WIDTH_CM, ARENA_HEIGHT_CM],
    [0.0,            ARENA_HEIGHT_CM]
], dtype=np.float32)

# --- Fire detection HSV range ---
FIRE_HSV_LOWER = np.array([0,  60, 60])
FIRE_HSV_UPPER = np.array([25, 255, 255])
FIRE_MIN_AREA  = 300

# --- Blue obstacle HSV range ---
#BLUE_HSV_LOWER = np.array([95, 50, 30])
#BLUE_HSV_UPPER = np.array([130, 255, 255])
BLUE_HSV_LOWER = np.array([100, 100,  50])
BLUE_HSV_UPPER = np.array([135, 255, 255])

# --- Obstacle detection ---
OBSTACLE_MIN_AREA         = 500
OBSTACLE_OVERLAY_ALPHA    = 0.4
OBSTACLE_PADDING_CELLS    = 3
OCCUPANCY_GRID_RESOLUTION = 5

# --- Camera ---
CAMERA_INDEX  = 0
FRAME_WIDTH   = 640
FRAME_HEIGHT  = 480
FRAME_FPS     = 30
