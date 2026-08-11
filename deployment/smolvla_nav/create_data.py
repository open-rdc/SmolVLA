#!/usr/bin/env python3
"""SmolVLA 学習用データの収集ノード（NavVLA deployment/scripts/create_data.py の SmolVLA 版）。

`/flag`(std_msgs/Empty) を1回publishするたびに収録の開始/停止がトグルし、
停止したところで1エピソードが確定する。NavVLA・VLA_nav・ml_planner と同じ作法。

    ros2 run smolvla_nav create_data
    ros2 topic pub --once /flag std_msgs/msg/Empty {}   # 開始
    ros2 topic pub --once /flag std_msgs/msg/Empty {}   # 停止（ここで1エピソード保存）

出力:

    <dataset_dir>/episode01/0.jpg 1.jpg ... N-1.jpg   224x224 BGR
                           /traj_data.pkl             {"position": (N,2) float32,
                                                       "yaw": (N,) float32 unwrap済}

**traj_prompt.txt は書かない**（NavVLA と同じ方針）。言語指示は収録後に
NavVLA の tools/lang_anotation_tool.py で付ける。同ツールは traj_prompt.txt が
無ければ jpg の枚数ぶんダミーで埋めてから編集させるので、そのまま使える。

    python ~/ros2_ws/src/NavVLA/tools/lang_anotation_tool.py <dataset_dir>

アノテーション後に変換器へ渡す:

    python training/data/lerobot_dataset.py --input <dataset_dir> --repo-id ... --root ...

（traj_prompt.txt が無いまま変換すると load_episode が落ちるので、
  アノテーション忘れはそこで気づける。）

NavVLA 版との違い:
  - 画像のデコードは cv_bridge ではなく navigation.py と同じ image_msg_to_bgr を使う。
    中央クロップも navigation.py と同一処理にして、収録時と推論時で前処理を揃える。
"""

from __future__ import annotations

import math
import os
import pickle
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_system_default
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from std_msgs.msg import Empty

from smolvla_nav.image_convert import image_msg_to_bgr


# 学習時と揃える定数（training/data/lerobot_dataset.py・navigation.py と一致させること）
FPS = 5
SAMPLE_INTERVAL = 1.0 / FPS          # 0.2s
IMG_H, IMG_W = 224, 224
JPEG_QUALITY = 95                    # NavVLA create_data.py と同じ


# colcon install 後は __file__ が site-packages 配下になり parents[2] ではリポジトリルートに
# 届かないため、env_humble.sh が設定する SMOLVLA_REPO_ROOT を優先する。未設定時（ソースツリーから
# 直接実行する場合）だけ従来通り __file__ から逆算する。
_env_repo_root = os.environ.get("SMOLVLA_REPO_ROOT")
REPO_ROOT = Path(_env_repo_root) if _env_repo_root else Path(__file__).resolve().parents[2]


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    """クォータニオンから yaw[rad] を取り出す（NavVLA create_data.py と同じ式）。"""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def center_crop_resize(bgr: np.ndarray) -> np.ndarray:
    """中央を正方形にクロップして 224x224 に縮小する。

    navigation.py の image_callback と同一処理にしてある（収録時と推論時で
    前処理がずれると、学習した見え方と実機の見え方が変わってしまうため）。
    """
    h, w = bgr.shape[:2]
    side = min(h, w)
    top = (h - side) // 2
    left = (w - side) // 2
    square = bgr[top : top + side, left : left + side]
    return cv2.resize(square, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)


class DataCreator(Node):
    def __init__(self) -> None:
        super().__init__("create_data")

        # --- パラメータ ---
        self.declare_parameter("odom_topic", "/Odometry")
        self.declare_parameter("image_topic", "/image_raw")
        # 既定の保存先は変換器がそのまま読める training/data/raw/ 配下。
        self.declare_parameter("output_dir", str(REPO_ROOT / "training" / "data" / "raw"))

        odom_topic = self.get_parameter("odom_topic").value
        image_topic = self.get_parameter("image_topic").value

        # --- 収録状態 ---
        self.collect_flag = False
        self.latest_odom: Optional[Odometry] = None
        self.latest_image: Optional[np.ndarray] = None      # BGR, フルサイズ

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.dataset_dir = Path(self.get_parameter("output_dir").value) / f"{timestamp}_dataset"
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

        self.current_episode_index = 0
        self.current_sample_index = 0
        self.total_collected_samples = 0
        self.current_episode_dir: Optional[Path] = None
        self.current_positions: list[list[float]] = []
        self.current_yaws: list[float] = []

        # --- 購読 ---
        self.create_subscription(Empty, "/flag", self.flag_callback, qos_profile_system_default)
        self.create_subscription(Odometry, odom_topic, self.odom_callback, qos_profile_system_default)
        self.create_subscription(Image, image_topic, self.image_callback, qos_profile_system_default)
        self.create_timer(SAMPLE_INTERVAL, self.timer_callback)

        self.get_logger().info(f"保存先: {self.dataset_dir}")
        self.get_logger().info(f"購読: {image_topic} / {odom_topic}")
        self.get_logger().info("/flag に Empty を publish すると収録の開始・停止がトグルします")
        self.get_logger().info("言語指示は収録後に lang_anotation_tool.py で付けてください")

    # ---- callbacks ----------------------------------------------------
    def flag_callback(self, _msg: Empty) -> None:
        self.collect_flag = not self.collect_flag
        if self.collect_flag:
            self._start_new_episode()
        else:
            self._finalize_current_episode()
            self.get_logger().info("🔴収録停止")

    def odom_callback(self, msg: Odometry) -> None:
        self.latest_odom = msg

    def image_callback(self, msg: Image) -> None:
        bgr = image_msg_to_bgr(msg)
        if bgr is None:
            self.get_logger().warn(
                f"未対応のencoding: {msg.encoding} (h={msg.height} w={msg.width} step={msg.step})",
                throttle_duration_sec=5.0,
            )
            return
        self.latest_image = bgr

    # ---- エピソード管理 ------------------------------------------------
    def _start_new_episode(self) -> None:
        self.current_episode_index += 1
        self.current_sample_index = 0
        self.current_positions = []
        self.current_yaws = []

        self.current_episode_dir = self.dataset_dir / f"episode{self.current_episode_index:02d}"
        self.current_episode_dir.mkdir(parents=True, exist_ok=True)
        self.get_logger().info(f"⚪収録開始: {self.current_episode_dir.name}")

    def _finalize_current_episode(self) -> None:
        """1エピソードを確定して traj_data.pkl を書く（言語指示は後からアノテーション）。"""
        if self.current_episode_dir is None:
            return

        if self.current_sample_index == 0:
            self.current_episode_dir.rmdir()
            self.current_episode_dir = None
            self.get_logger().info("🔴フレームが無いので空ディレクトリを削除しました")
            return

        positions = np.asarray(self.current_positions, dtype=np.float32)          # (N, 2)
        # yaw は ±pi で折り返すと差分(dyaw)が飛ぶので、エピソード全体で unwrap しておく。
        yaws = np.unwrap(np.asarray(self.current_yaws, dtype=np.float32)).astype(np.float32)

        with (self.current_episode_dir / "traj_data.pkl").open("wb") as f:
            pickle.dump({"position": positions, "yaw": yaws}, f)

        # traj_prompt.txt はここでは書かない（収録後に lang_anotation_tool.py で付ける）。
        assert self.current_sample_index == len(positions) == len(yaws)
        self.get_logger().info(
            f"🔵保存: {self.current_episode_dir.name} ({self.current_sample_index} フレーム)"
        )
        self.current_episode_dir = None

    # ---- サンプリング --------------------------------------------------
    def timer_callback(self) -> None:
        if not self.collect_flag or self.current_episode_dir is None:
            return
        if self.latest_odom is None or self.latest_image is None:
            self.get_logger().warn(
                f"データ待ち (odom={self.latest_odom is not None}, image={self.latest_image is not None})",
                throttle_duration_sec=2.0,
            )
            return

        # 画像と姿勢を同じタイミングでスナップショットする（参照代入はGIL下で原子的）。
        bgr = self.latest_image
        odom = self.latest_odom

        image_path = self.current_episode_dir / f"{self.current_sample_index}.jpg"
        cv2.imwrite(str(image_path), center_crop_resize(bgr), [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])

        pose = odom.pose.pose
        q = pose.orientation
        self.current_positions.append([pose.position.x, pose.position.y])
        self.current_yaws.append(yaw_from_quaternion(q.x, q.y, q.z, q.w))

        self.current_sample_index += 1
        self.total_collected_samples += 1
        if self.current_sample_index % FPS == 0:      # 1秒に1回だけログを出す
            self.get_logger().info(
                f"🟢{self.current_episode_dir.name} #{self.current_sample_index}"
            )

    # ---- 終了処理 ------------------------------------------------------
    def save_data(self) -> None:
        """Ctrl-C 時に、収録中のエピソードを取りこぼさず確定させる。"""
        if self.collect_flag:
            self._finalize_current_episode()
        if self.current_episode_index == 0:
            self.get_logger().info("🔴保存するデータがありません")
            return
        self.get_logger().info(
            f"🔵合計 {self.total_collected_samples} フレーム / "
            f"{self.current_episode_index} エピソードを {self.dataset_dir} に保存しました"
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DataCreator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Ctrl-C で終了します")
    finally:
        node.save_data()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
