"""SmolVLA本体 + トポロジカルマップ自己位置推定(place_prompt_node)をまとめて起動する。

SmolVLAはcolconパッケージ化していない(python3 script.pyで直接実行する構成)ため、
ament index経由ではなくファイルパス直接指定で起動する:

  conda activate ros_humble
  source ~/kasai_ws/env_humble.sh
  env -u PYTHONPATH ros2 launch ~/kasai_ws/src/SmolVLA/deployment/launch/smolvla_nav.launch.py

use_toponav:=false で place_prompt_node を止め、navigation.py の固定プロンプトのみで動かせる。
"""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration

DEPLOYMENT_DIR = Path(__file__).resolve().parents[1]


def generate_launch_description() -> LaunchDescription:
    use_toponav_arg = DeclareLaunchArgument(
        "use_toponav",
        default_value="true",
        description="true: place_prompt_node で自己位置推定して /prompt を自動更新する",
    )
    use_toponav = LaunchConfiguration("use_toponav")

    place_prompt_node = ExecuteProcess(
        cmd=["python3", "-u", str(DEPLOYMENT_DIR / "place_prompt_node.py")],
        cwd=str(DEPLOYMENT_DIR),
        output="screen",
        condition=IfCondition(use_toponav),
    )

    navigation_node = ExecuteProcess(
        cmd=["python3", "-u", str(DEPLOYMENT_DIR / "navigation.py")],
        cwd=str(DEPLOYMENT_DIR),
        output="screen",
    )

    return LaunchDescription(
        [
            use_toponav_arg,
            place_prompt_node,
            navigation_node,
        ]
    )
