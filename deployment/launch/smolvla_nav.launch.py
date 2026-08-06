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
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    use_toponav_arg = DeclareLaunchArgument(
        "use_toponav",
        default_value="true",
        description="true: place_prompt_node で自己位置推定して /prompt を自動更新する",
    )
    use_toponav = LaunchConfiguration("use_toponav")

    # --- 経路追従モード（既定は両方 OFF = 従来どおり生の速度をそのまま流す）---
    # 走行中に `ros2 param set /navigation <名前> <値>` でも切り替えられる。
    step_lookahead_arg = DeclareLaunchArgument(
        "step_lookahead",
        default_value="0",
        description="chunk の何ステップ先の行動を使うか。0=従来。不感帯の実測用に 5/10/15 と振る",
    )
    use_pure_pursuit_arg = DeclareLaunchArgument(
        "use_pure_pursuit",
        default_value="false",
        description="true: 操舵を Pure Pursuit に置き換える（並進速度はモデルの予測のまま）",
    )
    lookahead_distance_arg = DeclareLaunchArgument(
        "lookahead_distance",
        default_value="2.5",
        description="前方注視距離[m]。不感帯(約1〜2m)より長く取ること",
    )

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
        parameters=[
            {
                # LaunchConfiguration は文字列なので、宣言した型に明示的に変換する。
                "step_lookahead": ParameterValue(
                    LaunchConfiguration("step_lookahead"), value_type=int
                ),
                "use_pure_pursuit": ParameterValue(
                    LaunchConfiguration("use_pure_pursuit"), value_type=bool
                ),
                "lookahead_distance": ParameterValue(
                    LaunchConfiguration("lookahead_distance"), value_type=float
                ),
            }
        ],
    )

    return LaunchDescription(
        [
            use_toponav_arg,
            step_lookahead_arg,
            use_pure_pursuit_arg,
            lookahead_distance_arg,
            place_prompt_node,
            navigation_node,
        ]
    )
