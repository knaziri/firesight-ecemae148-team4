import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose2D, Point
from std_msgs.msg import String
import numpy as np
import time
import signal
import sys
from enum import Enum

from robot_pkg.vesc import VESC
from robot_pkg.config import (
    MAX_RPM,
    MIN_THROTTLE_FRAC,
    MAX_THROTTLE_FRAC,
    WATCHDOG_TIMEOUT_S,
)

# --- Navigation tuning ---
STOP_DISTANCE_CM       = 60.0
ARRIVAL_DISTANCE_CM    = 40.0
FULL_SPEED_DISTANCE_CM = 100.0

# --- PD gains ---
Kp = 0.7
Kd = 0.0


class RobotState(Enum):
    IDLE       = "IDLE"
    NAVIGATING = "NAVIGATING"
    ARRIVED    = "ARRIVED"


class NavNode(Node):

    def __init__(self):
        super().__init__('nav_node')

        self.vesc = VESC()

        signal.signal(signal.SIGINT,  self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self.create_subscription(
            Pose2D, '/overhead/robot_pose',
            self.pose_callback, 10)
        self.create_subscription(
            Point, '/overhead/target_waypoint',
            self.waypoint_callback, 10)

        self.state_pub = self.create_publisher(String, '/robot/state', 10)

        self.robot_pose         = None
        self.target_waypoint    = None
        self.state              = RobotState.IDLE
        self.last_msg_time      = time.time()
        self.last_heading_error = 0.0
        self.last_control_time  = time.time()

        self.create_timer(0.05, self.control_loop)
        self.create_timer(0.1,  self.watchdog)

        self.get_logger().info("Nav node started - waiting for waypoints.")

    # ── Signal handler ─────────────────────────────────────────────────────────

    def _signal_handler(self, sig, frame):
        self.get_logger().info(
            f"Signal {sig} received - neutralizing VESC")
        self._shutdown_vesc()
        sys.exit(0)

    def _shutdown_vesc(self):
        try:
            self.vesc.neutral()
            self.vesc.close()
        except Exception as e:
            self.get_logger().error(f"Error during VESC shutdown: {e}")

    # ── Callbacks ──────────────────────────────────────────────────────────────

    def pose_callback(self, msg):
        self.robot_pose    = (msg.x, msg.y, msg.theta)
        self.last_msg_time = time.time()

    def waypoint_callback(self, msg):
        self.last_msg_time = time.time()

        # Only accept new waypoints when IDLE
        # Once navigating ignore all updates so target doesn't shift
        if self.state != RobotState.IDLE:
            return

        self.target_waypoint    = (msg.x, msg.y)
        self.state              = RobotState.NAVIGATING
        self.last_heading_error = 0.0
        self.last_control_time  = time.time()
        self.get_logger().info(
            f"Waypoint accepted: "
            f"({msg.x:.1f}, {msg.y:.1f})cm - starting navigation"
        )

    # ── Watchdog ───────────────────────────────────────────────────────────────

    def watchdog(self):
        if time.time() - self.last_msg_time > WATCHDOG_TIMEOUT_S:
            if self.state == RobotState.NAVIGATING:
                self.get_logger().warn("Lost overhead signal - stopping")
                self.vesc.neutral()
                self.state              = RobotState.IDLE
                self.last_heading_error = 0.0
                self.publish_state()

    # ── Control loop ───────────────────────────────────────────────────────────

    def control_loop(self):
        if self.state != RobotState.NAVIGATING:
            return
        if self.robot_pose is None or self.target_waypoint is None:
            return

        rx, ry, heading = self.robot_pose
        tx, ty          = self.target_waypoint

        dx       = tx - rx
        dy       = ty - ry
        distance = np.sqrt(dx**2 + dy**2)

        # --- Arrived ---
        if distance < ARRIVAL_DISTANCE_CM:
            self.get_logger().info(f"Fire reached ({distance:.1f}cm)")
            self.vesc.neutral()
            self.state           = RobotState.ARRIVED
            self.target_waypoint = None
            self.publish_state()
            # Return to IDLE after pause so next waypoint can be accepted
            self.create_timer(2.0, self._return_to_idle)
            return

        # --- Slow stop zone ---
        if distance < STOP_DISTANCE_CM:
            self.vesc.set_steering(0.0)
            self.vesc.set_throttle_rpm(MIN_THROTTLE_FRAC)
            return

        # --- Heading error ---
        target_angle  = np.arctan2(dy, dx)
        heading_error = target_angle - heading
        heading_error = np.arctan2(
            np.sin(heading_error), np.cos(heading_error))

        # --- PD steering ---
        current_time = time.time()
        dt           = max(current_time - self.last_control_time, 1e-6)
        derivative   = (heading_error - self.last_heading_error) / dt
        steering     = (Kp * heading_error) + (Kd * derivative)
        steering     = float(np.clip(steering, -1.0, 1.0))

        self.last_heading_error = heading_error
        self.last_control_time  = current_time

        # --- Throttle via RPM ---
        distance_scale = min(1.0, distance / FULL_SPEED_DISTANCE_CM)
        throttle_frac  = MIN_THROTTLE_FRAC + distance_scale * \
                         (MAX_THROTTLE_FRAC - MIN_THROTTLE_FRAC)

        turn_reduction = 1.0 - 0.4 * abs(steering)
        throttle_frac  = float(np.clip(
            throttle_frac * turn_reduction,
            MIN_THROTTLE_FRAC,
            MAX_THROTTLE_FRAC
        ))

        # --- Send to VESC ---
        self.vesc.set_steering(steering)
        self.vesc.set_throttle_rpm(throttle_frac)

        self.get_logger().info(
            f"Dist: {distance:.1f}cm | "
            f"Heading err: {np.degrees(heading_error):.1f}deg | "
            f"Steering: {steering:.2f} | "
            f"Throttle: {throttle_frac:.2f} "
            f"({int(throttle_frac * MAX_RPM)} RPM)"
        )

        self.publish_state()

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _return_to_idle(self):
        if self.state == RobotState.ARRIVED:
            self.state = RobotState.IDLE
            self.get_logger().info(
                "Returned to IDLE - ready for next waypoint")
            self.publish_state()

    def publish_state(self):
        msg      = String()
        msg.data = self.state.value
        self.state_pub.publish(msg)

    def destroy_node(self):
        self.get_logger().info("Shutting down nav node.")
        self._shutdown_vesc()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = NavNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt received")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
