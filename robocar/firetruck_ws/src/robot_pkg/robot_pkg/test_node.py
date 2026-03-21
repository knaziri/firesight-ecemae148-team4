import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose2D, Point


class TestNode(Node):

    def __init__(self):
        super().__init__('test_node')

        # Subscribe to robot pose from overhead camera
        self.create_subscription(
            Pose2D,
            '/overhead/robot_pose',
            self.pose_callback,
            10
        )

        # Subscribe to target waypoint from overhead camera
        self.create_subscription(
            Point,
            '/overhead/target_waypoint',
            self.waypoint_callback,
            10
        )

        self.get_logger().info("Test node started - waiting for messages...")

    def pose_callback(self, msg):
        self.get_logger().info(
            f"Robot pose received: "
            f"x={msg.x:.1f}cm  y={msg.y:.1f}cm  "
            f"heading={msg.theta:.2f}rad"
        )

    def waypoint_callback(self, msg):
        self.get_logger().info(
            f"Waypoint received: "
            f"x={msg.x:.1f}cm  y={msg.y:.1f}cm"
        )


def main(args=None):
    rclpy.init(args=args)
    node = TestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
