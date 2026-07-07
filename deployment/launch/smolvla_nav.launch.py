"""SmolVLA本体 + トポロジカルマップ自己位置推定(place_prompt_node)をまとめて起動する。

  conda activate ros_humble
  source ~/kasai_ws/env_humble.sh
  env -u PYTHONPATH ros2 launch smolvla_nav smolvla_nav.launch.py

use_toponav:=false で place_prompt_node を止め、navigation.py の固定プロンプトのみで動かせる。
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    use_toponav_arg = DeclareLaunchArgument(
        "use_toponav",
        default_value="true",
        description="true: place_prompt_node で自己位置推定して /prompt を自動更新する",
    )
    use_toponav = LaunchConfiguration("use_toponav")

    place_prompt_node = Node(
        package="smolvla_nav",
        executable="place_prompt_node",
        name="place_prompt_node",
        output="screen",
        emulate_tty=True,
        condition=IfCondition(use_toponav),
    )

    navigation_node = Node(
        package="smolvla_nav",
        executable="navigation_node",
        name="navigation",
        output="screen",
        emulate_tty=True,
    )

    return LaunchDescription(
        [
            use_toponav_arg,
            place_prompt_node,
            navigation_node,
        ]
    )
