from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='robot_pkg',
            executable='nav_node',
            name='nav_node',
            output='screen',
            emulate_tty=True,
        ),
    ])
