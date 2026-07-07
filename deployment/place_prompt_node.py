#!/usr/bin/env python3
"""トポロジカルマップ上で自己位置推定し、対応する言語指示を /prompt に配信するノード。

navigation.py (SmolVLA) はすでに /image_raw(Image) と /prompt(String) を購読しているので、
このノードを navigation.py と並行起動するだけで、現在位置に応じた言語指示が自動的に切り替わる。

NavVLA の navvla/navigation.py の init_toponav / update_toponav_goal のパターンを踏襲。
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import yaml

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32, String

from toponav import TopologicalNavigator

THIS_DIR = Path(__file__).resolve().parent


def load_yaml(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def image_msg_to_bgr(msg: Image) -> Optional[np.ndarray]:
    channels = 3
    frame = np.frombuffer(msg.data, dtype=np.uint8)
    frame = frame.reshape(int(msg.height), int(msg.step))
    frame = frame[:, : int(msg.width) * channels]
    frame = frame.reshape(int(msg.height), int(msg.width), channels)

    encoding = msg.encoding.lower()
    if encoding == "bgr8":
        return frame
    if encoding == "rgb8":
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return None


class PlacePromptNode(Node):
    def __init__(self, config_path: Path) -> None:
        super().__init__("place_prompt_node")

        self.cfg = load_yaml(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        topomap_path = self._resolve(str(self.cfg.get("topomap_path", "config/topomap/topomap.yaml")))
        image_dir = self._resolve(str(self.cfg.get("topomap_image_dir", "config/topomap/images")))
        weight_path = self._resolve(str(self.cfg.get("placenet_weight_path", "weights/placenet.pt")))

        self.toponav_min_score = float(self.cfg.get("toponav_min_score", 0.4))

        self.toponav = TopologicalNavigator(
            topomap_path=topomap_path,
            image_dir=image_dir,
            weight_path=weight_path,
            device=self.device,
            image_size=(85, 85),
            crop_size=int(self.cfg.get("toponav_crop_size", 288)),
            delta=float(self.cfg.get("toponav_delta", 5.0)),
            window_lower=int(self.cfg.get("toponav_window_lower", -1)),
            window_upper=int(self.cfg.get("toponav_window_upper", 1)),
            window_radius=int(self.cfg.get("toponav_window_radius", 2)),
        )
        self.get_logger().info(f"Topomap loaded: nodes={len(self.toponav.nodes)} path={topomap_path}")

        self.latest_bgr: Optional[np.ndarray] = None
        self.last_published_instruction: Optional[str] = None
        self.toponav_current_index: Optional[int] = None

        self.image_sub = self.create_subscription(Image, "/image_raw", self.image_callback, 10)
        self.prompt_pub = self.create_publisher(String, "/prompt", 10)
        self.current_node_pub = self.create_publisher(Int32, "/toponav/current_node", 10)

        interval_ms = int(self.cfg.get("interval_ms", 100))
        self.timer = self.create_timer(interval_ms / 1000.0, self.timer_callback)
        self._last_log_t: Optional[float] = None

    def _resolve(self, raw_path: str) -> Path:
        path = Path(raw_path)
        return path if path.is_absolute() else THIS_DIR / path

    def image_callback(self, msg: Image) -> None:
        bgr = image_msg_to_bgr(msg)
        if bgr is None:
            self.get_logger().warn(f"unsupported encoding: {msg.encoding}", throttle_duration_sec=5.0)
            return
        self.latest_bgr = bgr

    def timer_callback(self) -> None:
        obs_bgr = self.latest_bgr
        if obs_bgr is None:
            return

        current_index, score = self.toponav.estimate_current_node(obs_bgr)
        current_node = self.toponav.nodes[current_index]
        self.current_node_pub.publish(Int32(data=int(current_node.node_id)))

        below = score < self.toponav_min_score
        self.get_logger().info(
            f"[toponav] current_id={current_node.node_id} instruction=\"{current_node.instruction}\" "
            f"wmass={score:.3f} thr={self.toponav_min_score:.3f}{' BELOW' if below else ''}",
            throttle_duration_sec=1.0,
        )

        # 信頼度が低いときは instruction の切り替えを見送る（推定値のpublishは上で実施済み）
        if below:
            return
        if current_index == self.toponav_current_index:
            return
        self.toponav_current_index = current_index

        if current_node.instruction == self.last_published_instruction:
            return
        self.last_published_instruction = current_node.instruction
        self.prompt_pub.publish(String(data=current_node.instruction))
        self.get_logger().info(f"[toponav] /prompt <- \"{current_node.instruction}\"")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(THIS_DIR / "config" / "topomap_nav.yaml"),
        help="topomap_nav.yaml のパス",
    )
    parsed = parser.parse_args()

    rclpy.init()
    node = PlacePromptNode(Path(parsed.config))
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
