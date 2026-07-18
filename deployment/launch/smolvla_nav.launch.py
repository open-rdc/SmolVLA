"""SmolVLA本体 + トポロジカルマップ自己位置推定(place_prompt_node)を起動する。

  source <ROS2ワークスペース>/env_humble.sh   # conda ros_humble (Python3.12) に切り替え
  ros2 launch smolvla_nav smolvla_nav.launch.py

use_toponav:=false で place_prompt_node を止め、navigation.py の固定プロンプトのみで動かせる。

カメラはこのlaunchでは起動しない。/image_raw は icart_driver 側の usb_cam_node が配信する
(smolvla_navとicart_driverで別々にカメラを持つと同じデバイスを取り合って衝突するため)。
navigation単体でテストしたい場合は、別途 v4l2_camera_node 等を手動起動すること。
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
