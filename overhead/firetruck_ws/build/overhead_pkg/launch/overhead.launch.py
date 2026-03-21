from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='overhead_pkg',
            executable='overhead_node',
            name='overhead_node',
            output='screen',
            emulate_tty=True,
        )
    ])
