import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose2D, Point, PoseArray, Pose
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import MapMetaData
from std_msgs.msg import Header
import cv2
import numpy as np
import time

from overhead_pkg.calibration        import try_compute_homography
from overhead_pkg.fire_detection     import detect_fires
from overhead_pkg.obstacle_detection import (
    build_obstacle_mask,
    get_obstacle_contours,
    build_occupancy_grid,
    draw_obstacles,
    draw_arena_boundary,
)
from overhead_pkg.robot_detection    import detect_robot, draw_robot
from overhead_pkg.measurements       import compute_paths, draw_paths
from overhead_pkg.pathfinding        import grid_to_world
from overhead_pkg.utils              import build_arena_mask
from overhead_pkg.fire_detection     import draw_fires
from overhead_pkg.config             import (
    CAMERA_INDEX,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    FRAME_FPS,
    OCCUPANCY_GRID_RESOLUTION,
    ARENA_WIDTH_CM,
    ARENA_HEIGHT_CM,
)


REPLAN_DISTANCE_CM    = 5.0
REPLAN_OBSTACLE_DELTA = 10


class OverheadNode(Node):

    def __init__(self):
        super().__init__('overhead_node')

        # ── Publishers ──────────────────────────────────────────────────────
        self.robot_pub    = self.create_publisher(
            Pose2D,        '/overhead/robot_pose',        10)
        self.fire_pub     = self.create_publisher(
            PoseArray,     '/overhead/fire_positions',    10)
        self.waypoint_pub = self.create_publisher(
            Point,         '/overhead/target_waypoint',   10)
        self.grid_pub     = self.create_publisher(
            OccupancyGrid, '/overhead/occupancy_grid',    10)

        # ── Camera ──────────────────────────────────────────────────────────
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS,          FRAME_FPS)

        # Lock exposure to prevent auto adjustment in changing light
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, -8)
        self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)

        if not self.cap.isOpened():
            self.get_logger().error("Failed to open camera.")
            raise RuntimeError("Camera not available")

        # ── ArUco ────────────────────────────────────────────────────────────
        self.aruco_dict   = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters_create()

        # Tuned for glare conditions
        self.aruco_params.adaptiveThreshWinSizeMin  = 3
        self.aruco_params.adaptiveThreshWinSizeMax  = 53
        self.aruco_params.adaptiveThreshWinSizeStep = 10
        self.aruco_params.adaptiveThreshConstant    = 7

        # ── Navigation state ─────────────────────────────────────────────────
        self.H              = None
        self.H_inv          = None
        self.grid           = None
        self.paths          = []
        self.last_robot_pos = None
        self.last_grid      = None
        self.last_fires     = []

        # ── Fire persistence ─────────────────────────────────────────────────
        self.fire_detection_count = 0
        self.fire_absence_count   = 0
        self.locked_fire_pos      = None
        self.LOCK_FRAMES          = 8
        self.UNLOCK_FRAMES        = 15

        # ── Timer - 10hz ─────────────────────────────────────────────────────
        self.timer = self.create_timer(0.1, self.process_frame)
        self.get_logger().info("Overhead node started.")

    # ── Main callback ──────────────────────────────────────────────────────────

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("Failed to grab frame.")
            return

        display = frame.copy()
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- ArUco detection ---
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.aruco_params)

        # --- Dynamic calibration ---
        new_H, new_H_inv = try_compute_homography(corners, ids)
        if new_H is not None:
            self.H     = new_H
            self.H_inv = new_H_inv
            self.get_logger().debug("Homography updated.")

        # --- Always draw markers and boundary ---
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(display, corners, ids)
        draw_arena_boundary(display, corners, ids)

        if self.H is None:
            # Show calibration status before markers are found
            found_ids = ids.flatten().tolist() if ids is not None else []
            from overhead_pkg.config import CORNER_IDS
            cv2.putText(display,
                "Waiting for corner markers (IDs 0, 1, 2, 3)...",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
            for i, cid in enumerate(CORNER_IDS):
                color  = (0, 255, 0) if cid in found_ids else (0, 0, 255)
                status = "OK" if cid in found_ids else "missing"
                cv2.putText(display, f"Corner ID {cid}: {status}",
                    (10, 60 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.imshow("Overhead Debug", display)
            cv2.waitKey(1)
            return

        # --- Arena mask ---
        arena_mask = build_arena_mask(corners, ids, frame.shape)
        if arena_mask is None:
            arena_mask = self._build_full_frame_mask(frame.shape)

        # --- Obstacle detection ---
        obstacle_mask     = build_obstacle_mask(frame, arena_mask)
        obstacle_contours = get_obstacle_contours(obstacle_mask)
        self.grid         = build_occupancy_grid(obstacle_mask, self.H)

        # --- Robot and fire detection ---
        robot = detect_robot(corners, ids, self.H)
        fires = detect_fires(frame, self.H)

        # --- Publish ---
        self._publish_robot(robot)
        self._publish_fires(fires)
        self._publish_occupancy_grid(self.grid)

        # --- Path planning ---
        self._update_paths(robot, fires)

        # --- Waypoint with persistence lock ---
        self._publish_waypoint(robot, fires)

        # --- Draw overlays ---
        draw_obstacles(display, obstacle_contours)
        draw_robot(display, robot)
        draw_fires(display, fires)
        draw_paths(display, self.paths, fires)

        # Draw locked fire position if active
        if self.locked_fire_pos is not None:
            pt = np.array([[[self.locked_fire_pos[0],
                              self.locked_fire_pos[1]]]], dtype=np.float32)
            px = cv2.perspectiveTransform(pt, self.H_inv)
            lx = int(px[0][0][0])
            ly = int(px[0][0][1])
            cv2.circle(display, (lx, ly), 12, (0, 255, 0), 3)
            cv2.putText(display, "LOCKED",
                (lx + 15, ly),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # --- Calibration status ---
        cal_status = "Calibration: LIVE" if new_H is not None \
                     else "Calibration: LOST (cached)"
        cal_color  = (0, 255, 0) if new_H is not None else (0, 165, 255)
        cv2.putText(display, cal_status,
            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, cal_color, 2)

        # --- Status bar ---
        lock_status = f"LOCKED ({self.locked_fire_pos[0]:.0f}," \
                      f"{self.locked_fire_pos[1]:.0f})" \
                      if self.locked_fire_pos else \
                      f"detecting {self.fire_detection_count}/{self.LOCK_FRAMES}"
        cv2.putText(display,
            f"Fires: {len(fires)}  |  "
            f"Fire lock: {lock_status}  |  "
            f"Obstacles: {len(obstacle_contours)}  |  "
            f"Robot: {'detected' if robot else 'not detected'}",
            (10, display.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 2)

        cv2.imshow("Overhead Debug", display)
        cv2.waitKey(1)

        self.get_logger().info(
            f"Fires: {len(fires)} | "
            f"Robot: {'detected' if robot else 'not detected'} | "
            f"Lock: {'LOCKED' if self.locked_fire_pos else 'searching'}"
        )

    # ── Publish helpers ────────────────────────────────────────────────────────

    def _publish_robot(self, robot):
        if robot is None:
            return
        msg       = Pose2D()
        msg.x     = float(robot['wx'])
        msg.y     = float(robot['wy'])
        msg.theta = float(robot['heading'])
        self.robot_pub.publish(msg)

    def _publish_fires(self, fires):
        if not fires:
            return
        msg                 = PoseArray()
        msg.header          = Header()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        for fire in fires:
            p            = Pose()
            p.position.x = float(fire['wx'])
            p.position.y = float(fire['wy'])
            p.position.z = 0.0
            msg.poses.append(p)
        self.fire_pub.publish(msg)

    def _publish_occupancy_grid(self, grid):
        if grid is None:
            return
        msg                 = OccupancyGrid()
        msg.header          = Header()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.info            = MapMetaData()
        msg.info.resolution = float(OCCUPANCY_GRID_RESOLUTION) / 100.0
        msg.info.width      = int(grid.shape[1])
        msg.info.height     = int(grid.shape[0])
        msg.info.origin     = Pose()
        msg.data            = (grid.flatten() * 100).tolist()
        self.grid_pub.publish(msg)

    def _publish_waypoint(self, robot, fires):
        if robot is None:
            return

        if fires:
            # Fire visible this frame
            self.fire_absence_count = 0

            if self.locked_fire_pos is None:
                # Not yet locked - increment detection counter
                self.fire_detection_count += 1
                self.get_logger().info(
                    f"Fire detected {self.fire_detection_count}/"
                    f"{self.LOCK_FRAMES} frames before lock"
                )
                if self.fire_detection_count >= self.LOCK_FRAMES:
                    self.locked_fire_pos      = (
                        fires[0]['wx'], fires[0]['wy'])
                    self.fire_detection_count = 0
                    self.get_logger().info(
                        f"Fire position LOCKED: "
                        f"({self.locked_fire_pos[0]:.1f}, "
                        f"{self.locked_fire_pos[1]:.1f})cm"
                    )
            # Already locked - do not update position

        else:
            # Fire not visible this frame
            self.fire_detection_count = 0

            if self.locked_fire_pos is not None:
                self.fire_absence_count += 1
                self.get_logger().info(
                    f"Fire absent {self.fire_absence_count}/"
                    f"{self.UNLOCK_FRAMES} frames before unlock"
                )
                if self.fire_absence_count >= self.UNLOCK_FRAMES:
                    self.get_logger().info("Fire lost - releasing lock")
                    self.locked_fire_pos    = None
                    self.fire_absence_count = 0

        # Only publish if locked
        if self.locked_fire_pos is None:
            return

        wp     = Point()
        wp.x   = float(self.locked_fire_pos[0])
        wp.y   = float(self.locked_fire_pos[1])
        wp.z   = 0.0
        self.waypoint_pub.publish(wp)

    # ── Path planning ──────────────────────────────────────────────────────────

    def _update_paths(self, robot, fires):
        if robot is None or not fires or self.grid is None:
            return
        if self._should_replan(robot, fires):
            self.paths          = compute_paths(robot, fires, self.grid,
                                                self.H_inv)
            self.last_robot_pos = (robot['wx'], robot['wy'])
            self.last_grid      = self.grid.copy()
            self.last_fires     = list(fires)
            self.get_logger().info("Replanned paths.")

    def _should_replan(self, robot, fires):
        if len(self.paths) == 0:
            return True
        if self._robot_moved(robot):
            return True
        if self._obstacles_changed():
            return True
        if self._fires_changed(fires):
            return True
        return False

    def _robot_moved(self, robot):
        if self.last_robot_pos is None:
            return True
        dx = robot['wx'] - self.last_robot_pos[0]
        dy = robot['wy'] - self.last_robot_pos[1]
        return np.sqrt(dx**2 + dy**2) > REPLAN_DISTANCE_CM

    def _obstacles_changed(self):
        if self.last_grid is None or self.grid is None:
            return True
        if self.last_grid.shape != self.grid.shape:
            return True
        delta = int(np.sum(np.abs(
            self.grid.astype(int) - self.last_grid.astype(int))))
        return delta > REPLAN_OBSTACLE_DELTA

    def _fires_changed(self, fires):
        if len(fires) != len(self.last_fires):
            return True
        for a, b in zip(fires, self.last_fires):
            if abs(a['wx'] - b['wx']) > REPLAN_DISTANCE_CM or \
               abs(a['wy'] - b['wy']) > REPLAN_DISTANCE_CM:
                return True
        return False

    # ── Utility ────────────────────────────────────────────────────────────────

    def _build_full_frame_mask(self, frame_shape):
        mask       = np.zeros(frame_shape[:2], dtype=np.uint8)
        mask[:]    = 255
        return mask

    def destroy_node(self):
        self.get_logger().info("Shutting down overhead node.")
        self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


# ── Entry point ────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = OverheadNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
