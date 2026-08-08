"""SmolVLA本体 + トポロジカルマップ自己位置推定(place_prompt_node) + 経路追従(path_follower_node)を起動する。

  source <ROS2ワークスペース>/env_humble.sh   # conda ros_humble (Python3.12) に切り替え
  ros2 launch smolvla_nav smolvla_nav.launch.py

use_toponav:=false で place_prompt_node を止め、navigation.py の固定プロンプトのみで動かせる。

カメラはこのlaunchでは起動しない。/image_raw は icart_driver 側の usb_cam_node が配信する
(smolvla_navとicart_driverで別々にカメラを持つと同じデバイスを取り合って衝突するため)。
navigation単体でテストしたい場合は、別途 v4l2_camera_node 等を手動起動すること。

navigation_node は /cmd_vel を直接 publish しない。v とフォールバック用の生 dyaw を
/smolvla_cmd_vel_raw に出し、path_follower_node が /smolvla_pred_path を Pure Pursuit で
追従して最終的な /cmd_vel を publish する（決定: pred_path が正しく cmd_vel に反映され
ていなかったバグ修正で、操舵の決定を別ノードに分離した）。
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

    # --- navigation_node 側（速度計算・不感帯の実測用）---
    step_lookahead_arg = DeclareLaunchArgument(
        "step_lookahead",
        default_value="0",
        description="chunk の何ステップ先の行動をフォールバック操舵に使うか。0=従来",
    )

    # --- path_follower_node 側（経路追従）---
    use_pure_pursuit_arg = DeclareLaunchArgument(
        "use_pure_pursuit",
        default_value="true",
        description="true: /smolvla_pred_path を Pure Pursuit で追従して操舵する。"
        "false: navigation_node の生dyawをそのまま使う",
    )
    lookahead_distance_arg = DeclareLaunchArgument(
        "lookahead_distance",
        default_value="2.5",
        description="前方注視距離[m]。不感帯(約1〜2m)より長く取ること",
    )
    path_timeout_sec_arg = DeclareLaunchArgument(
        "path_timeout_sec",
        default_value="5.0",
        description="/smolvla_pred_path がこれより古ければ生dyawにフォールバックする[s]。"
        "navigation_node の [latency] ログの infer_chunk 時間より大きくすること",
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
            }
        ],
    )

    path_follower_node = Node(
        package="smolvla_nav",
        executable="path_follower_node",
        name="path_follower",
        output="screen",
        emulate_tty=True,
        parameters=[
            {
                "use_pure_pursuit": ParameterValue(
                    LaunchConfiguration("use_pure_pursuit"), value_type=bool
                ),
                "lookahead_distance": ParameterValue(
                    LaunchConfiguration("lookahead_distance"), value_type=float
                ),
                "path_timeout_sec": ParameterValue(
                    LaunchConfiguration("path_timeout_sec"), value_type=float
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
            path_timeout_sec_arg,
            place_prompt_node,
            navigation_node,
            path_follower_node,
        ]
    )
