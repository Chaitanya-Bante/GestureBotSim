from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='turtlesim',
            namespace='turtlesim',
            executable='turtlesim_node',
            name='sim'
        ),
        Node(
            package='gesture_control',
            namespace='GestureBotSim',
            executable='turtle_control',
            name='sim'
        )
    ])