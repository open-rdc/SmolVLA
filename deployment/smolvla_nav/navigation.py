#!/usr/bin/env python3
"""SmolVLA navigation inference node.

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

import threading

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image

from smolvla_nav.image_convert import image_msg_to_bgr
from std_msgs.msg import Bool, String


# 学習時と揃える定数（training/data/lerobot_dataset.py と一致させること）
FPS = 5
DT = 1.0 / FPS
IMG_H, IMG_W = 224, 224

# チェックポイントの場所（tar.gz を展開した先）
DEFAULT_CKPT = Path(__file__).resolve().parents[2] / "training" / "data" / "weight" / "smolvla_nav201_ns_scratch_ckpt"


# ══════════════════════════════════════════════════════════════════
#  SmolVLA ラッパー
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

    @torch.no_grad()
    def infer_chunk(self, image_rgb: np.ndarray, state: np.ndarray, task: str) -> np.ndarray:
        """1 枚の観測から 50 ステップ分の行動列をまとめて返す（非同期先読み用）。

        select_action は内部キューから1手ずつ返すが、こちらは chunk 全体を返すので
        呼び出し側で自前キューを管理できる。

        Returns:
            actions: np.ndarray shape (chunk_size, 2) = [[dx_body, dyaw], ...]
        """
        img = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
        batch = {
            "observation.images.front": img,
            "observation.state": torch.from_numpy(np.asarray(state, np.float32)),
            "task": task,
        }
        batch = self.preprocessor(batch)
        if self.device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                chunk = self.policy.predict_action_chunk(batch)  # (1, chunk_size, action_dim)
        else:
            chunk = self.policy.predict_action_chunk(batch)
        chunk = self.postprocessor(chunk)
        return chunk.squeeze(0).float().cpu().numpy()  # (chunk_size, 2)


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

        # --- パラメータ（速度上限・制御周期）---
        self.linear_max_vel = 1.0
        self.angular_max_vel = 0.8
        self.interval_ms = 200                 # 制御周期 = DT(200ms) と揃える

        # --- 非同期パラメータ g（SmolVLA論文 3.3節）---
        # 残量が g*n を下回ったら次chunkを推論する（g=0.7 = 30%消費で再推論、論文推奨）。
        self.g = 0.7
        self.chunk_size = int(self.model.policy.config.chunk_size)   # = 50
        self.refill_threshold = int(self.g * self.chunk_size)        # = 35

        # 重複区間の集約関数（lerobot async_inference の weighted_average と同じ）。
        # 新旧chunkの同じ時刻の行動を 0.2*旧 + 0.8*新 で混ぜて滑らかに繋ぐ。
        self.aggregate_fn = lambda old, new: 0.2 * old + 0.8 * new

        # --- 非同期用: 絶対タイムステップ付きの行動辞書 + Lock ---
        # lerobot 同様、各行動を絶対時刻(step)で管理する。推論スレッドは新chunkを
        # 時刻でそろえて既存キューに集約し、制御スレッドは現在stepの行動を取り出す。
        self._step = 0                              # 次に実行する行動の絶対index
        self._actions: dict[int, np.ndarray] = {}   # {step: action(2,)}
        self._queue_lock = threading.Lock()

        # --- 購読と publish ---
        self.image_sub = self.create_subscription(Image, "/image_raw", self.image_callback, 10)
        self.autonomous_sub = self.create_subscription(Bool, "/autonomous", self.autonomous_callback, 10)
        self.prompt_sub = self.create_subscription(String, "/prompt", self.prompt_callback, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        # --- タイマーを別々のコールバックグループに分ける ---
        # MultiThreadedExecutor と併用し、重い推論(~1.2s)が制御ループを止めないようにする。
        # 各グループは MutuallyExclusive なので、推論の二重起動も防げる。
        self._control_group = MutuallyExclusiveCallbackGroup()
        self._infer_group = MutuallyExclusiveCallbackGroup()
        self.control_timer = self.create_timer(
            self.interval_ms / 1000.0, self.control_timer_callback, callback_group=self._control_group
        )
        self.infer_timer = self.create_timer(
            self.interval_ms / 1000.0, self.inference_timer_callback, callback_group=self._infer_group
        )


    # ---- callbacks ----------------------------------------------------
    def autonomous_callback(self, msg: Bool) -> None:
        # 自律 OFF -> ON の立ち上がりで行動キューをリセット（前エピソードの残りを捨てる）。
        if msg.data and not self.autonomous_flag:
            self.model.reset()
            with self._queue_lock:
                self._actions.clear()
                self._step = 0
        self.autonomous_flag = msg.data

    def prompt_callback(self, msg: String) -> None:
        self.latest_prompt = msg.data

    def image_callback(self, msg: Image) -> None:
        # ROS Image -> numpy(HWC) -> RGB -> 中央正方形クロップ -> 224x224 -> self.latest_image
        bgr = image_msg_to_bgr(msg)
        if bgr is None:
            self.get_logger().warn(
                f"unsupported encoding: {msg.encoding} (h={msg.height} w={msg.width} step={msg.step})",
                throttle_duration_sec=5.0,
            )
            return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # 3) 中央を正方形にクロップしてから 224x224 に縮小（学習時と同じ形にする）。
        h, w = rgb.shape[:2]
        side = min(h, w)
        top = (h - side) // 2
        left = (w - side) // 2
        square = rgb[top : top + side, left : left + side]
        resized = cv2.resize(square, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)

        # infer() が期待する HWC・uint8・RGB・(224,224,3) の形で保存。
        self.latest_image = np.ascontiguousarray(resized)


    # ---- 制御ループ（軽量・絶対に止めない）---------------------------
    def control_timer_callback(self) -> None:
        """200ms ごとに現在 step の行動を取り出して cmd_vel を発行する。推論はしない。"""
        if not self.autonomous_flag:
            return  # 非自律時は publish しない（他コントローラに任せる）

        # 現在 step の行動を取り出し、時刻を1つ進める（行動が無くても step は進める）。
        with self._queue_lock:
            action = self._actions.pop(self._step, None)
            self._step += 1

        if action is None:
            # まだ chunk が用意できていない（起動直後など）→ 安全のため停止指令。
            self.cmd_vel_pub.publish(Twist())
            return

        # action[dx_body, dyaw] を dt で割って速度にし、上限でクリップして発行。
        dx_body, dyaw = float(action[0]), float(action[1])
        v = float(np.clip(dx_body / DT, -self.linear_max_vel, self.linear_max_vel))
        omega = float(np.clip(dyaw / DT, -self.angular_max_vel, self.angular_max_vel))
        cmd_vel = Twist()
        cmd_vel.linear.x = v
        cmd_vel.angular.z = omega
        self.cmd_vel_pub.publish(cmd_vel)

    # ---- 推論ループ（重い・別スレッド）-------------------------------
    def inference_timer_callback(self) -> None:
        """残量が g*n(=refill_threshold) を下回ったら最新観測で chunk を計算し、
        絶対時刻でそろえて既存キューに集約する（lerobot _aggregate_action_queues 相当）。

        SmolVLA論文 3.3節の g=0.7（30%消費で再推論）。MutuallyExclusiveCallbackGroup
        なので ~1.2s ブロックしても制御タイマーは別スレッドで回り続け、二重起動もしない。
        """
        if not self.autonomous_flag or self.latest_image is None:
            return

        with self._queue_lock:
            remaining = sum(1 for ts in self._actions if ts >= self._step)
            base_step = self._step   # この観測が予測する行動列の起点となる絶対時刻
        if remaining >= self.refill_threshold:
            return  # 残量 >= g*n なのでまだ再推論しない

        # 参照代入は GIL 下で原子的なので、最新値をスナップショットして使う。
        image = self.latest_image
        prompt = self.latest_prompt
        # state = [v, omega]。暫定ゼロ固定（copycat 対策で現在指令は入れない）。
        state = np.zeros(2, dtype=np.float32)

        # ここが重い（~1.2s）。制御ループとは別スレッドなので停止しない。
        # chunk[i] は絶対時刻 base_step + i の行動に対応する。
        chunk = self.model.infer_chunk(image, state, prompt)  # (chunk_size, 2)

        # 絶対時刻でそろえて集約（lerobot と同じロジック）:
        #  - 既に実行済み(ts < 現在step)は捨てる（推論中に経過したぶん）
        #  - 未来で既存に無い時刻はそのまま追加
        #  - 既存にある時刻(=重複区間)は aggregate_fn(旧,新)=0.2旧+0.8新 で混ぜて滑らかに繋ぐ
        with self._queue_lock:
            cur_step = self._step
            for i, a in enumerate(chunk):
                ts = base_step + i
                if ts < cur_step:
                    continue
                if ts in self._actions:
                    self._actions[ts] = self.aggregate_fn(self._actions[ts], a)
                else:
                    self._actions[ts] = a


def main() -> int:
    rclpy.init()
    node = SmolVLANavigationNode()
    # 制御タイマー・推論タイマー・I/O を別スレッドで並列に回す。
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
