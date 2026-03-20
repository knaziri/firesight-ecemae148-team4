import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import time

from robot_pkg.vesc import VESC


class TestVescNode(Node):
    """
    Tests both current control and RPM control modes.
    Start with current control since it works without motor detection.
    """

    def __init__(self):
        super().__init__('test_vesc_node')

        self.vesc = VESC()

        self.state_pub  = self.create_publisher(String, '/robot/state', 10)
        self.step       = 0
        self.start_time = time.time()

        self.create_timer(0.1, self.run_sequence)
        self.get_logger().info("Test VESC node started.")

    def publish_state(self, state: str):
        msg      = String()
        msg.data = state
        self.state_pub.publish(msg)
        self.get_logger().info(state)

    def run_sequence(self):
        elapsed = time.time() - self.start_time

        # 0-2s: center steering, no throttle
        if elapsed < 2.0 and self.step == 0:
            self.publish_state("STEP 1: Centering steering, no throttle")
            self.vesc.set_steering(0.0)
            self.vesc.stop()
            self.step = 1

        # 2-4s: try current control forward
        elif elapsed >= 2.0 and self.step == 1:
            self.publish_state("STEP 2: Current control forward (2A)")
            self.vesc.set_steering(0.0)
            self.vesc.set_throttle_current(0.4)  # 40% of MAX_CURRENT = 2A
            self.step = 2

        # 4-6s: try RPM control forward
        elif elapsed >= 4.0 and self.step == 2:
            self.publish_state("STEP 3: RPM control forward (1000 RPM)")
            self.vesc.set_steering(0.0)
            self.vesc.set_throttle_rpm(0.33)  # 33% of MAX_RPM = ~1000 RPM
            self.step = 3

        # 6-8s: steer left with current control
        elif elapsed >= 6.0 and self.step == 3:
            self.publish_state("STEP 4: Steer left")
            self.vesc.set_steering(-0.5)
            self.vesc.set_throttle_current(0.4)
            self.step = 4

        # 8-10s: steer right with current control
        elif elapsed >= 8.0 and self.step == 4:
            self.publish_state("STEP 5: Steer right")
            self.vesc.set_steering(0.5)
            self.vesc.set_throttle_current(0.4)
            self.step = 5

        # 10s+: stop and print telemetry
        elif elapsed >= 10.0 and self.step == 5:
            self.publish_state("STEP 6: Stopping")
            self.vesc.stop()
            self.step = 6

        elif self.step == 6:
            telemetry = self.vesc.get_telemetry()
            if telemetry:
                self.get_logger().info(
                    f"Voltage: {telemetry['voltage']:.1f}V | "
                    f"RPM: {telemetry['rpm']} | "
                    f"Current: {telemetry['current']:.1f}A | "
                    f"Temp: {telemetry['temp_fet']:.1f}C"
                )
            else:
                self.get_logger().warn(
                    "Could not read telemetry - "
                    "check UART is enabled in VESC Tool"
                )
            self.publish_state("SEQUENCE COMPLETE")
            self.step = 7

    def destroy_node(self):
        self.vesc.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TestVescNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
