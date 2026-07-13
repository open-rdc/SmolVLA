#!/usr/bin/env python3
"""SmolVLA navigation inference node.

NavVLA の deployment/navvla/navigation.py を参考にした SmolVLA 版の推論スクリプト。
OmniVLA との違い（このチェックポイントの仕様）:

  - 入力画像は front カメラ 1 枚だけ（preprocessor が front -> camera1 に rename する）
  - observation.state = [v, omega]   (2 次元, body frame の並進速度と角速度。常にゼロ固定)
  - task = 言語指示の文字列（内部で tokenize される）
  - 出力 action = [dx, dy, hx, hy]   (chunk_size=50 個のwaypoint列。全waypointが
    「推論時点の現在姿勢」を共通原点とした絶対オフセット。hx,hy = cos/sin(heading)。
    NavVLA形式で、SmolVLA標準の「前ステップからの連鎖差分」ではない点に注意)
  - 学習は FPS=5 => dt=0.2s。
  - v, omega への変換はPure Pursuit（経路追従制御）で行う。経路上のルックアヘッド点への
    曲率から omega を、直近waypoint間隔から v を求める（control_timer_callback参照）。
    フィードバックは経過時間ベースのオープンループ（実測姿勢は使わない）。

正規化・rename・tokenize・waypointの原点変換(WaypointRebase/Unscale)は保存済みの
preprocessor/postprocessorが自動でやってくれるので、こちらが用意するのは
「生の」観測dictだけでよい。
"""

from __future__ import annotations

import math
import time
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
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Path as NavPath
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
        """1 ステップ推論して action [dx, dy, hx, hy] を返す（このnodeでは未使用。参考用）。

        Args:
            image_rgb: front カメラ画像。HWC, uint8, RGB。(IMG_H, IMG_W にリサイズ済み想定)
            state:     [v, omega] の float 配列 (shape (2,))。
            task:      言語指示（例: "go straight along the road"）。

        Returns:
            action: np.ndarray shape (4,) = [dx, dy, hx, hy]（逆正規化済みの実スケール）。
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

        return action.squeeze(0).float().cpu().numpy()  # (4,) = [dx, dy, hx, hy]

    @torch.no_grad()
    def infer_chunk(self, image_rgb: np.ndarray, state: np.ndarray, task: str) -> np.ndarray:
        """1 枚の観測から chunk_size 個ぶんのwaypoint列をまとめて返す。

        全waypointが「この推論時点の現在姿勢」を共通原点とした絶対オフセット
        [dx, dy, hx, hy]（hx,hy = cos/sin(heading)）。前ステップからの連鎖差分ではない。

        Returns:
            actions: np.ndarray shape (chunk_size, 4) = [[dx, dy, hx, hy], ...]
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
        return chunk.squeeze(0).float().cpu().numpy()  # (chunk_size, 4)


# ══════════════════════════════════════════════════════════════════
#  Pure Pursuit 幾何計算（ROS非依存、単体テスト可能）
# ══════════════════════════════════════════════════════════════════
def _interpolate_pose(chunk: np.ndarray, k: float) -> tuple[float, float, float]:
    """chunk内の実数index kにおける姿勢(x, y, theta)を線形補間して返す。

    chunk: shape (N, 4) = [[dx, dy, hx, hy], ...]（全waypointが共通原点の絶対値）。
    k はクリップされる（0 <= k <= N-1）。
    """
    n = len(chunk)
    k = max(0.0, min(k, n - 1))
    i0 = int(math.floor(k))
    i1 = min(i0 + 1, n - 1)
    frac = k - i0
    x = float(chunk[i0, 0] * (1 - frac) + chunk[i1, 0] * frac)
    y = float(chunk[i0, 1] * (1 - frac) + chunk[i1, 1] * frac)
    cos_ = chunk[i0, 2] * (1 - frac) + chunk[i1, 2] * frac
    sin_ = chunk[i0, 3] * (1 - frac) + chunk[i1, 3] * frac
    theta = math.atan2(float(sin_), float(cos_))
    return x, y, theta


def _find_lookahead_point(
    chunk: np.ndarray, start_idx: int, x_now: float, y_now: float, lookahead_distance: float
) -> tuple[float, float]:
    """(x_now, y_now) からの距離が lookahead_distance 以上になる最初のwaypointを
    chunk[start_idx:] から探して返す。見つからなければ最終点にフォールバックする。
    """
    n = len(chunk)
    start_idx = max(0, min(start_idx, n - 1))
    for idx in range(start_idx, n):
        dx = float(chunk[idx, 0]) - x_now
        dy = float(chunk[idx, 1]) - y_now
        if math.hypot(dx, dy) >= lookahead_distance:
            return float(chunk[idx, 0]), float(chunk[idx, 1])
    return float(chunk[-1, 0]), float(chunk[-1, 1])


def pure_pursuit_command(
    chunk: np.ndarray, elapsed: float, dt: float, lookahead_distance: float
) -> tuple[float, float]:
    """経過時間 elapsed における (v, omega) をPure Pursuitで計算する（クリップ前の生値）。

    「今どこにいるか」はchunk生成からの経過時間でchunk内を補間して推定する
    （オープンループ、実測姿勢は使わない）。omegaはルックアヘッド点への曲率
    （経路の先の形状を見て早めに曲がる）、vは直近waypoint間隔（モデル自身が
    意図しているペースを尊重する）から求める。

    Args:
        chunk: shape (N, 4) = [[dx, dy, hx, hy], ...]（共通原点の絶対waypoint、N>=2）。
        elapsed: chunk生成からの経過時間 [s]。
        dt: 1ステップの時間間隔 [s]。
        lookahead_distance: ルックアヘッド距離 [m]。

    Returns:
        (v, omega): クリップ前の速度指令。
    """
    n = len(chunk)
    k = elapsed / dt
    x_now, y_now, theta_now = _interpolate_pose(chunk, k)

    k0 = max(0, min(int(math.floor(k)), n - 2))
    # v: 直近waypoint間隔(chunk[k0]->chunk[k0+1])から、モデル自身が意図しているペースを求める。
    pace_dx = float(chunk[k0 + 1, 0]) - float(chunk[k0, 0])
    pace_dy = float(chunk[k0 + 1, 1]) - float(chunk[k0, 1])
    v = math.hypot(pace_dx, pace_dy) / dt

    # omega: ルックアヘッド点への曲率から求める（標準Pure Pursuit公式）。
    x_l, y_l = _find_lookahead_point(chunk, k0 + 1, x_now, y_now, lookahead_distance)
    dxg, dyg = x_l - x_now, y_l - y_now
    x_local = math.cos(theta_now) * dxg + math.sin(theta_now) * dyg
    y_local = -math.sin(theta_now) * dxg + math.cos(theta_now) * dyg
    d = math.hypot(x_local, y_local)
    if d < 1e-6:
        omega = 0.0
    else:
        curvature = 2.0 * y_local / (d * d)
        omega = v * curvature

    return v, omega


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

        # --- パラメータ（速度上限・制御周期・Pure Pursuit）---
        self.linear_max_vel = 1.0
        self.angular_max_vel = 1.0
        self.interval_ms = 200                 # 制御周期 = DT(200ms) と揃える
        self.lookahead_distance = 0.5          # Pure Pursuitのルックアヘッド距離 [m]

        # --- 非同期再推論パラメータ g（SmolVLA論文 3.3節）---
        # chunk生成からの経過時間が (1-g)*chunk_size*DT を超えたら次chunkを推論する
        # （g=0.7 = 30%消費相当で再推論、論文推奨）。カデンスは旧実装（残量ベース）と数値的に同じ。
        self.g = 0.7
        self.chunk_size = int(self.model.policy.config.chunk_size)   # = 50
        self.refill_after_sec = (1.0 - self.g) * self.chunk_size * DT  # 約3秒

        # --- 最新chunkの単一スロット + Lock ---
        # 各waypointは推論時点の現在姿勢を共通原点とした絶対オフセットなので、
        # 旧実装のような絶対時刻indexキュー(dict)は不要。最新chunkを丸ごと保持し、
        # 制御ループは経過時間からPure Pursuitでv,omegaを都度計算する（control_timer_callback参照）。
        self._chunk: Optional[np.ndarray] = None     # shape (chunk_size, 4) = [[dx,dy,hx,hy], ...]
        self._chunk_time: Optional[float] = None      # time.monotonic() 時刻（chunk生成時）
        self._chunk_lock = threading.Lock()

        # --- 経路可視化用: 直近に推論したchunk(今後10秒の予測)をそのままPathとして publish ---
        # 過去は蓄積せず、新しいchunkが来るたびに置き換える。base_link基準（=常に現在位置からの相対軌跡）。
        self.path_frame_id = "base_link"

        # --- 購読と publish ---
        self.image_sub = self.create_subscription(Image, "/image_raw", self.image_callback, 10)
        self.autonomous_sub = self.create_subscription(Bool, "/autonomous", self.autonomous_callback, 10)
        self.prompt_sub = self.create_subscription(String, "/prompt", self.prompt_callback, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.pred_path_pub = self.create_publisher(NavPath, "/smolvla_pred_path", 10)

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
        # 自律 OFF -> ON の立ち上がりでchunkをリセット（前エピソードの残りを捨てる）。
        if msg.data and not self.autonomous_flag:
            self.model.reset()
            with self._chunk_lock:
                self._chunk = None
                self._chunk_time = None
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
        """200ms ごとにPure Pursuitでv,omegaを計算してcmd_velを発行する。推論はしない。"""
        if not self.autonomous_flag:
            return  # 非自律時は publish しない（他コントローラに任せる）

        with self._chunk_lock:
            chunk = self._chunk
            chunk_time = self._chunk_time

        if chunk is None or chunk_time is None:
            # まだ chunk が用意できていない（起動直後など）→ 安全のため停止指令。
            self.cmd_vel_pub.publish(Twist())
            return

        # chunk生成からの経過時間で「今どこにいるはずか」を推定し(オープンループ)、
        # Pure Pursuitでv,omegaを求める。詳細は pure_pursuit_command() 参照。
        elapsed = time.monotonic() - chunk_time
        v_raw, omega_raw = pure_pursuit_command(chunk, elapsed, DT, self.lookahead_distance)

        v = float(np.clip(v_raw, -self.linear_max_vel, self.linear_max_vel))
        omega = float(np.clip(omega_raw, -self.angular_max_vel, self.angular_max_vel))
        cmd_vel = Twist()
        cmd_vel.linear.x = v
        cmd_vel.angular.z = omega
        self.cmd_vel_pub.publish(cmd_vel)

    def _publish_pred_path(self, chunk: np.ndarray) -> None:
        """直近に推論したchunk（今後10秒分の予測）をそのまま base_link 基準の Path として publish。

        各waypointが既に「今」を共通原点とした絶対値なので積分は不要（そのままplotするだけ）。
        過去は蓄積しない。新しいchunkが来るたびに置き換え（毎回 poses を作り直す）。
        base_link 基準 = 常に「現在のロボット位置」からの相対軌跡として解釈される。
        """
        stamp = self.get_clock().now().to_msg()
        poses: list[PoseStamped] = []
        for dx, dy, hx, hy in chunk:
            pose = PoseStamped()
            pose.header.stamp = stamp
            pose.header.frame_id = self.path_frame_id
            pose.pose.position.x = float(dx)
            pose.pose.position.y = float(dy)
            theta = math.atan2(float(hy), float(hx))
            pose.pose.orientation.z = math.sin(theta / 2.0)
            pose.pose.orientation.w = math.cos(theta / 2.0)
            poses.append(pose)

        path_msg = NavPath()
        path_msg.header.stamp = stamp
        path_msg.header.frame_id = self.path_frame_id
        path_msg.poses = poses
        self.pred_path_pub.publish(path_msg)

    # ---- 推論ループ（重い・別スレッド）-------------------------------
    def inference_timer_callback(self) -> None:
        """chunk生成からの経過時間が refill_after_sec を超えたら最新観測でchunkを再計算する。

        SmolVLA論文 3.3節のg=0.7相当（カデンスは旧実装の残量ベース判定と数値的に同じ、
        約3秒ごと）。MutuallyExclusiveCallbackGroupなので ~1.2s ブロックしても
        制御タイマーは別スレッドで回り続け、二重起動もしない。
        """
        if not self.autonomous_flag or self.latest_image is None:
            return

        with self._chunk_lock:
            chunk_time = self._chunk_time
        if chunk_time is not None and (time.monotonic() - chunk_time) < self.refill_after_sec:
            return  # まだ再推論しない

        # 参照代入は GIL 下で原子的なので、最新値をスナップショットして使う。
        # t_capture はこの画像に対応する時刻（新chunkのt=0に対応、推論の~1.2sを含めない）。
        t_capture = time.monotonic()
        image = self.latest_image
        prompt = self.latest_prompt
        # state = [v, omega]。暫定ゼロ固定（copycat 対策で現在指令は入れない）。
        state = np.zeros(2, dtype=np.float32)

        # ここが重い（~1.2s）。制御ループとは別スレッドなので停止しない。
        chunk = self.model.infer_chunk(image, state, prompt)  # (chunk_size, 4)

        # 今回のchunk(=今後10秒の予測、クリップ前の生値)をそのままRViz可視化用に publish。
        self._publish_pred_path(chunk)

        with self._chunk_lock:
            self._chunk = chunk
            self._chunk_time = t_capture


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
