import rclpy
from rclpy.node import Node
import time

from robot_pkg.vesc import VESC
from robot_pkg.config import MAX_RPM


# --- Test parameters - tune these for your setup ---
TEST_RPM_FRAC   = 0.33    # fraction of MAX_RPM to use during test
                           # 0.15 * 3000 = 450 RPM - slow and safe
STEP_DURATION_S = 3.0     # seconds per step


class TestRpmSteeringNode(Node):
    """
    Tests RPM throttle and steering simultaneously.

    Sequence:
        1. Straight forward (RPM, steering=0.0)
        2. Forward + steer left
        3. Forward + steer right
        4. Forward + center steering
        5. Stop
    """

    def __init__(self):
        super().__init__('test_rpm_steering_node')

        self.vesc       = VESC()
        self.step       = 0
        self.start_time = time.time()
        self.rpm        = int(TEST_RPM_FRAC * MAX_RPM)

        self.create_timer(0.1, self.run_sequence)

        self.get_logger().info(
            f"RPM + Steering test started. "
            f"Using {self.rpm} RPM ({TEST_RPM_FRAC * 100:.0f}% of max). "
            f"Each step lasts {STEP_DURATION_S}s."
        )

    def run_sequence(self):
        elapsed = time.time() - self.start_time

        # Step 1: 0-3s straight forward
        if elapsed < STEP_DURATION_S and self.step == 0:
            self.get_logger().info(
                f"STEP 1: Straight forward | "
                f"RPM={self.rpm} | Steering=0.0")
            self.vesc.set_steering(0.0)
            self.vesc.set_throttle_rpm(TEST_RPM_FRAC)
            self.step = 1

        # Step 2: 3-6s forward + steer left
        elif elapsed >= STEP_DURATION_S and self.step == 1:
            self.get_logger().info(
                f"STEP 2: Forward + steer LEFT | "
                f"RPM={self.rpm} | Steering=-0.5")
            self.vesc.set_steering(-0.5)
            self.vesc.set_throttle_rpm(TEST_RPM_FRAC)
            self.step = 2

        # Step 3: 6-9s forward + steer right
        elif elapsed >= STEP_DURATION_S * 2 and self.step == 2:
            self.get_logger().info(
                f"STEP 3: Forward + steer RIGHT | "
                f"RPM={self.rpm} | Steering=0.5")
            self.vesc.set_steering(0.5)
            self.vesc.set_throttle_rpm(TEST_RPM_FRAC)
            self.step = 3

        # Step 4: 9-12s forward + center steering
        elif elapsed >= STEP_DURATION_S * 3 and self.step == 3:
            self.get_logger().info(
                f"STEP 4: Forward + center | "
                f"RPM={self.rpm} | Steering=0.0")
            self.vesc.set_steering(0.0)
            self.vesc.set_throttle_rpm(TEST_RPM_FRAC)
            self.step = 4

        # Step 5: 12s+ stop
        elif elapsed >= STEP_DURATION_S * 4 and self.step == 4:
            self.get_logger().info("STEP 5: Stopping")
            self.vesc.neutral()
            self.step = 5

        elif self.step == 5:
            self.get_logger().info(
                "Sequence complete. "
                "If wheels spun during steps 2 and 3 while steering moved "
                "simultaneously, RPM + steering control is confirmed working."
            )
            self.step = 6

    def destroy_node(self):
        self.get_logger().info("Shutting down test node.")
        self.vesc.neutral()
        self.vesc.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TestRpmSteeringNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
