#!/usr/bin/env python3
"""SmolVLA navigation inference node (雛形).

NavVLA の deployment/navvla/navigation.py を参考にした SmolVLA 版の推論スクリプト。
OmniVLA との違い（このチェックポイントの仕様）:

  - 入力画像は front カメラ 1 枚だけ（preprocessor が front -> camera1 に rename する）
  - observation.state = [v, omega]   (2 次元, body frame の並進速度と角速度)
  - task = 言語指示の文字列（内部で tokenize される）
  - 出力 action = [dx_body, dyaw]    (chunk_size=50 の行動列を内部キューで管理)
  - 学習は FPS=5 => dt=0.2s。よって v = dx_body/dt, omega = dyaw/dt。

正規化(mean/std)・rename・tokenize は保存済みの preprocessor/postprocessor が
自動でやってくれるので、こちらが用意するのは「生の」観測 dict だけでよい。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

import rclpy  
from rclpy.node import Node  
from geometry_msgs.msg import Twist  
from sensor_msgs.msg import Image  
from std_msgs.msg import Bool, String 


# 学習時と揃える定数（training/data/lerobot_dataset.py と一致させること）
FPS = 5
DT = 1.0 / FPS
IMG_H, IMG_W = 224, 224

# チェックポイントの場所（tar.gz を展開した先）
DEFAULT_CKPT = Path(__file__).resolve().parents[1] / "training" / "data" / "weight" / "smolvla_tsudanuma_ckpt"


# ══════════════════════════════════════════════════════════════════
#  SmolVLA ラッパー  ―― ここは実装済み。中身を読んで理解するのが目的。
# ══════════════════════════════════════════════════════════════════
class SmolVLAModel:
    """学習済み SmolVLA をロードし、1 枚の画像+状態+指示から action を返す。"""

    def __init__(self, ckpt_dir: Path = DEFAULT_CKPT, device: Optional[str] = None) -> None:
        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))

        # 1) ポリシー本体（重み込み）をロード。config.json / model.safetensors を読む。
        self.policy = SmolVLAPolicy.from_pretrained(str(ckpt_dir))
        self.policy.to(self.device).eval()

        # 2) 保存済みの前処理/後処理パイプラインをロード。
        #    preprocessor : rename(front->camera1) -> batch化 -> tokenize -> device転送 -> 正規化
        #    postprocessor: action の逆正規化（mean/std を戻す）
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=str(ckpt_dir),
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )

        # 行動キューを空に。エピソード（自律走行）を開始するたびに reset() を呼ぶ。
        self.policy.reset()

    def reset(self) -> None:
        """自律走行を開始/再開するたびに呼ぶ（内部の action chunk キューを空にする）。"""
        self.policy.reset()

    @torch.no_grad()
    def infer(self, image_rgb: np.ndarray, state: np.ndarray, task: str) -> np.ndarray:
        """1 ステップ推論して action [dx_body, dyaw] を返す。

        Args:
            image_rgb: front カメラ画像。HWC, uint8, RGB。(IMG_H, IMG_W にリサイズ済み想定)
            state:     [v, omega] の float 配列 (shape (2,))。
            task:      言語指示（例: "go straight along the road"）。

        Returns:
            action: np.ndarray shape (2,) = [dx_body, dyaw]（逆正規化済みの実スケール）。
        """
        # HWC uint8 [0,255] -> CHW float [0,1]。VISUAL は IDENTITY 正規化なので
        # [0,1] で渡す（内部の prepare_images が [-1,1] に変換する）。
        img = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0

        batch = {
            "observation.images.front": img,                                  # (C,H,W)
            "observation.state": torch.from_numpy(np.asarray(state, np.float32)),  # (2,)
            "task": task,
        }

        batch = self.preprocessor(batch)          # 正規化・tokenize・device 転送
        # GPU では fp16 autocast で推論（Turing のテンソルコアで約3倍速、精度低下は実質なし）。
        # 誤差の出やすい演算は autocast が自動で fp32 に保つ。CPU 時は従来どおり fp32。
        if self.device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                action = self.policy.select_action(batch)  # (1, action_dim) 内部キューから1手
        else:
            action = self.policy.select_action(batch)
        action = self.postprocessor(action)        # 逆正規化して実スケールへ

        return action.squeeze(0).float().cpu().numpy()  # (2,) = [dx_body, dyaw]


# ══════════════════════════════════════════════════════════════════
#  ROS2 ノード
# ══════════════════════════════════════════════════════════════════


class SmolVLANavigationNode(Node):
    def __init__(self) -> None:
        super().__init__("smolvla_navigation")

        # --- モデル（実装済み）---
        self.model = SmolVLAModel()
        self.get_logger().info("SmolVLA loaded.")

        # --- 状態変数 ---
        self.autonomous_flag = False
        self.latest_image: Optional[np.ndarray] = None   # RGB, (IMG_H, IMG_W, 3)
        self.latest_prompt = "go straight along the road"
        self.state = np.zeros(2, dtype=np.float32)        # [v, omega]

        # --- パラメータ（速度上限・推論周期）---
        self.linear_max_vel = 0.3
        self.angular_max_vel = 0.3
        self.interval_ms = 200

        # --- 購読と publish ---
        self.image_sub = self.create_subscription(Image, "/image_raw", self.image_callback, 10)
        self.autonomous_sub = self.create_subscription(Bool, "/autonomous", self.autonomous_callback, 10)
        self.prompt_sub = self.create_subscription(String, "/prompt", self.prompt_callback, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.timer = self.create_timer(self.interval_ms / 1000.0, self.policy_timer_callback)


    # ---- callbacks ----------------------------------------------------
    def autonomous_callback(self, msg: Bool) -> None:
        # 自律 OFF -> ON の立ち上がりで行動キューをリセット（前エピソードの残りを捨てる）。
        if msg.data and not self.autonomous_flag:
            self.model.reset()
        self.autonomous_flag = msg.data

    def prompt_callback(self, msg: String) -> None:
        self.latest_prompt = msg.data

    def image_callback(self, msg: Image) -> None:
        # ROS Image -> numpy(HWC) -> RGB -> 中央正方形クロップ -> 224x224 -> self.latest_image
        # 1) 生バイト列を HxWxC の配列に戻す（cv_bridge 非依存。rgb8/bgr8 のみ対応）。
        channels = 3
        frame = np.frombuffer(msg.data, dtype=np.uint8)
        frame = frame.reshape(int(msg.height), int(msg.step))            # step にはパディングが含まれ得る
        frame = frame[:, : int(msg.width) * channels]                   # 余分なパディング列を捨てる
        frame = frame.reshape(int(msg.height), int(msg.width), channels)

        # 2) encoding を見て RGB に統一する（学習データは RGB だった）。
        encoding = msg.encoding.lower()
        if encoding == "bgr8":
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif encoding == "rgb8":
            rgb = frame
        else:
            self.get_logger().warn(f"unsupported encoding: {msg.encoding}", throttle_duration_sec=5.0)
            return

        # 3) 中央を正方形にクロップしてから 224x224 に縮小（学習時と同じ形にする）。
        h, w = rgb.shape[:2]
        side = min(h, w)
        top = (h - side) // 2
        left = (w - side) // 2
        square = rgb[top : top + side, left : left + side]
        resized = cv2.resize(square, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)

        # infer() が期待する HWC・uint8・RGB・(224,224,3) の形で保存。
        self.latest_image = np.ascontiguousarray(resized)


    # ---- main loop ----------------------------------------------------
    def policy_timer_callback(self) -> None:
        if not self.autonomous_flag or self.latest_image is None:
            return

        # 1) state = [v, omega]。学習時は「1 つ前の増分/dt」（copycat 対策で現在指令は入れない）。
        #    暫定でゼロ固定。将来 odom から前ステップの実速度を入れると精度が上がる。
        self.state = np.zeros(2, dtype=np.float32)

        # 2) 推論
        action = self.model.infer(self.latest_image, self.state, self.latest_prompt)
        dx_body, dyaw = float(action[0]), float(action[1])

        # 3) action -> cmd_vel 変換。増分 [dx_body, dyaw] を dt で割って速度にし、上限でクリップ。
        v = dx_body / DT
        omega = dyaw / DT
        v = float(np.clip(v, -self.linear_max_vel, self.linear_max_vel))
        omega = float(np.clip(omega, -self.angular_max_vel, self.angular_max_vel))
        cmd_vel = Twist()
        cmd_vel.linear.x = v
        cmd_vel.angular.z = omega
        self.cmd_vel_pub.publish(cmd_vel)


def main() -> int:
    rclpy.init()
    node = SmolVLANavigationNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
