#!/usr/bin/env python3
"""SmolVLA navigation inference node.

NavVLA の deployment/navvla/navigation.py を参考にした SmolVLA 版の推論スクリプト。
OmniVLA との違い（このチェックポイントの仕様）:

  - 入力画像は同一カメラの時系列 3 枚（camera1=現在 / camera2=1秒前 / camera3=2秒前）。
    SmolVLA の 3 視点スロットを時間軸に転用しているので front キーは使わない。
  - observation.state = [v, omega]   (2 次元, body frame の並進速度と角速度)
  - task = 言語指示の文字列（内部で tokenize される）
  - 出力 action = [dx_body, dyaw]    (chunk_size=50 の行動列を内部キューで管理)
  - 学習は FPS=5 => dt=0.2s。よって v = dx_body/dt, omega = dyaw/dt。

正規化(mean/std)・rename・tokenize は保存済みの preprocessor/postprocessor が
自動でやってくれるので、こちらが用意するのは「生の」観測 dict だけでよい。
"""

from __future__ import annotations

import collections
import math
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
from geometry_msgs.msg import Twist, PoseStamped, PointStamped
from nav_msgs.msg import Path as NavPath
from sensor_msgs.msg import Image

from smolvla_nav.image_convert import image_msg_to_bgr
from smolvla_nav.pure_pursuit import find_lookahead_point, integrate_path, pure_pursuit_omega
from std_msgs.msg import Bool, String


# 学習時と揃える定数（training/data/lerobot_dataset.py と一致させること）
FPS = 5
DT = 1.0 / FPS
IMG_H, IMG_W = 224, 224

# 時系列画像コンテキストのラグ。5 フレーム @ FPS=5 = 1 秒。
# 学習側 training/data/lerobot_dataset.py の HISTORY_STRIDE_FRAMES と必ず一致させる。
# 設計: ~/.company/engineering/docs/smolvla-temporal-context-architecture.md 決定3
HISTORY_STRIDE_FRAMES = 5
HISTORY_LEN = 2 * HISTORY_STRIDE_FRAMES + 1   # 2秒分 = 11 枚

# --- 経路追従モードの既定値（既定は両方 OFF = 従来どおり生の速度をそのまま流す）---
# 実機で「コース端に寄ると中央に戻れない」問題の切り分け/対策用。モデルの計画は
# 「1〜2秒まっすぐ → その後ゆるやかに戻る」形をしているが、推論が ~1.2s かかるため
# chunk の先頭6ステップ程度しか実行されず、曲がり始める部分に到達しない。
# 実験中に ros2 param set で切り替えたいので ROS2 パラメータにしてある。
DEFAULT_STEP_LOOKAHEAD = 0          # chunk の何ステップ先の行動を使うか（不感帯の実測用）
DEFAULT_USE_PURE_PURSUIT = False    # True で操舵だけ Pure Pursuit に置き換える
DEFAULT_LOOKAHEAD_DISTANCE = 2.5    # 前方注視距離 [m]。不感帯(約1〜2m)より長く取ること
PP_MAX_PATH_STEPS = 50              # Pure Pursuit 用に積分する最大ステップ数（=chunk長）

# チェックポイントの場所（tar.gz を展開した先）
DEFAULT_CKPT = Path(__file__).resolve().parents[2] / "training" / "data" / "weight" / "smolvla_all_30ep_ckpt" / "pretrained_model"


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

    def _build_batch(self, images_rgb: list[np.ndarray], state: np.ndarray, task: str) -> dict:
        """時系列3枚+状態+指示から観測 dict を組み立てる（正規化前の「生の」batch）。

        camera1=現在(t) / camera2=1秒前 / camera3=2秒前 の順に入れる。学習時の
        build_features() と同じ割り当てにすること。
        VISUAL は IDENTITY 正規化なので [0,1] で渡す（内部の prepare_images が [-1,1] に変換する）。
        """
        if len(images_rgb) != 3:
            raise ValueError(f"images_rgb must have 3 frames (t, t-1s, t-2s), got {len(images_rgb)}")

        batch: dict = {
            # HWC uint8 [0,255] -> CHW float [0,1]
            f"observation.images.camera{i + 1}": torch.from_numpy(im).permute(2, 0, 1).float() / 255.0
            for i, im in enumerate(images_rgb)
        }
        batch["observation.state"] = torch.from_numpy(np.asarray(state, np.float32))  # (2,)
        batch["task"] = task
        return batch

    @torch.no_grad()
    def infer(self, images_rgb: list[np.ndarray], state: np.ndarray, task: str) -> np.ndarray:
        """1 ステップ推論して action [dx_body, dyaw] を返す。

        Args:
            images_rgb: [現在(t), 1つ前(t-1), 2つ前(t-2)] の3枚。各 HWC, uint8, RGB。
                        (IMG_H, IMG_W にリサイズ済み想定)
            state:     [v, omega] の float 配列 (shape (2,))。
            task:      言語指示（例: "go straight along the road"）。

        Returns:
            action: np.ndarray shape (2,) = [dx_body, dyaw]（逆正規化済みの実スケール）。
        """
        batch = self._build_batch(images_rgb, state, task)
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
    def infer_chunk(self, images_rgb: list[np.ndarray], state: np.ndarray, task: str) -> np.ndarray:
        """3枚の時系列観測から 50 ステップ分の行動列をまとめて返す（非同期先読み用）。

        select_action は内部キューから1手ずつ返すが、こちらは chunk 全体を返すので
        呼び出し側で自前キューを管理できる。

        Args:
            images_rgb: [現在(t), 1つ前(t-1), 2つ前(t-2)] の3枚。infer() と同じ形式。

        Returns:
            actions: np.ndarray shape (chunk_size, 2) = [[dx_body, dyaw], ...]
        """
        batch = self._build_batch(images_rgb, state, task)
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

        # --- 画像履歴バッファ（2秒分=11枚、最新が右端）---
        # 制御周期(DT=200ms)ごとに最新フレームを push する。推論の呼び出し頻度
        # （レイテンシ依存で変動する）ではなく、学習データの生フレーム間隔 dt=0.2s と
        # 揃えるためにこの周期でサンプリングする。
        # 取り出し時: 現在=history[-1], 1秒前=history[-1-5], 2秒前=history[-1-10]
        # 設計: ~/.company/engineering/docs/smolvla-temporal-context-architecture.md 決定3
        self.image_history: collections.deque[np.ndarray] = collections.deque(maxlen=HISTORY_LEN)
        self._history_lock = threading.Lock()

        # --- パラメータ（速度上限・制御周期）---
        self.linear_max_vel = 1.0
        self.angular_max_vel = 1.0
        self.interval_ms = 200                 # 制御周期 = DT(200ms) と揃える

        # --- 経路追従モード（復帰動作の実験用。既定は両方 OFF = 従来の挙動）---
        # step_lookahead: chunk の K ステップ先の行動を使う。計画の先頭にある
        #   「まっすぐな前置き（不感帯）」を飛ばせるので、K を 0→5→10→15 と振ると
        #   不感帯の長さがステップ数で実測できる。0 = 従来どおり現在stepの行動。
        # use_pure_pursuit: True で ω を Pure Pursuit で計算する。v は従来どおり
        #   モデルの予測を使い、操舵だけを置き換える（カーブの挙動を壊さないため）。
        # lookahead_distance: 前方注視距離[m]。不感帯より短いと不感帯の中を見てしまい
        #   生の dyaw を使うのと変わらなくなるので 2〜3m から始める。
        self.declare_parameter("step_lookahead", DEFAULT_STEP_LOOKAHEAD)
        self.declare_parameter("use_pure_pursuit", DEFAULT_USE_PURE_PURSUIT)
        self.declare_parameter("lookahead_distance", DEFAULT_LOOKAHEAD_DISTANCE)

        # 重複区間の集約関数（lerobot async_inference の weighted_average と同じ）。
        # 新旧chunkの同じ時刻の行動を 0.2*旧 + 0.8*新 で混ぜて滑らかに繋ぐ。
        self.aggregate_fn = lambda old, new: 0.2 * old + 0.8 * new

        # --- 非同期用: 絶対タイムステップ付きの行動辞書 + Lock ---
        # lerobot 同様、各行動を絶対時刻(step)で管理する。推論スレッドは新chunkを
        # 時刻でそろえて既存キューに集約し、制御スレッドは現在stepの行動を取り出す。
        self._step = 0                              # 次に実行する行動の絶対index
        self._actions: dict[int, np.ndarray] = {}   # {step: action(2,)}
        self._queue_lock = threading.Lock()

        # --- 経路可視化用: 直近に推論したchunk(今後10秒の予測)をそのままPathとして publish ---
        # 過去は蓄積せず、新しいchunkが来るたびに置き換える。base_link基準（=常に現在位置からの相対軌跡）。
        self.path_frame_id = "base_link"

        # --- 購読と publish ---
        self.image_sub = self.create_subscription(Image, "/image_raw", self.image_callback, 10)
        self.autonomous_sub = self.create_subscription(Bool, "/autonomous", self.autonomous_callback, 10)
        self.prompt_sub = self.create_subscription(String, "/prompt", self.prompt_callback, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.pred_path_pub = self.create_publisher(NavPath, "/smolvla_pred_path", 10)
        # 注視点も出す。/smolvla_pred_path と重ねて見ると、注視点が計画の
        # 「まっすぐな前置き」の中に入っていないか（=注視距離が短すぎないか）を目視できる。
        self.lookahead_pub = self.create_publisher(PointStamped, "/smolvla_lookahead", 10)

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
            # 前回走行の画像履歴を持ち越さない（行動キューのクリアと同じ理由）。
            with self._history_lock:
                self.image_history.clear()
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

        # 学習データのフレーム間隔(dt=0.2s)と揃えるため、推論の呼び出し頻度ではなく
        # この制御ループの周期で画像履歴をサンプリングする（決定3参照）。
        image = self.latest_image     # 参照代入は GIL 下で原子的
        if image is not None:
            with self._history_lock:
                self.image_history.append(image)

        # 走行中に ros2 param set で切り替えられるよう毎tick読み直す（5Hzなので負荷は無視できる）。
        step_lookahead = max(int(self.get_parameter("step_lookahead").value), 0)
        use_pure_pursuit = bool(self.get_parameter("use_pure_pursuit").value)
        lookahead_distance = float(self.get_parameter("lookahead_distance").value)

        # 現在 step の行動を取り出し、時刻を1つ進める（行動が無くても step は進める）。
        with self._queue_lock:
            step = self._step
            current = self._actions.get(step)      # 「今」の行動（並進速度はこれを使う）
            selected = current                     # 実際に操舵に使う行動
            if step_lookahead:
                # K ステップ先の行動を使う。まだ届いていなければ現在stepにフォールバック。
                ahead = self._actions.get(step + step_lookahead)
                if ahead is not None:
                    selected = ahead
            # Pure Pursuit 用に、現在stepから連続して存在する行動を集める（キューは変えない）。
            path_actions: list[np.ndarray] = []
            if use_pure_pursuit:
                s = step
                while len(path_actions) < PP_MAX_PATH_STEPS and s in self._actions:
                    path_actions.append(self._actions[s])
                    s += 1
            self._actions.pop(step, None)          # 消化（辞書が無限に伸びないように）
            self._step += 1

        if current is None and selected is None:
            # まだ chunk が用意できていない（起動直後など）→ 安全のため停止指令。
            self.cmd_vel_pub.publish(Twist())
            return

        # 並進速度は常にモデルの「今」の予測から取る。Pure Pursuit でも v は置き換えない
        # （速度側は既に期待どおり動いているので、操舵だけを差し替える）。
        speed_src = current if current is not None else selected
        v = float(np.clip(float(speed_src[0]) / DT, -self.linear_max_vel, self.linear_max_vel))

        omega: Optional[float] = None
        if use_pure_pursuit and len(path_actions) >= 2:
            # 計画を積分して経路にし、前方注視点に向かう円弧の角速度を求める。
            # 「計画の初速(dyaw[0])」ではなく「計画の行き先」を見るので、計画先頭の
            # 不感帯（まっすぐな前置き）を飛び越せる。
            poses = integrate_path(np.asarray(path_actions, dtype=np.float64))
            goal = find_lookahead_point(poses, lookahead_distance)
            if goal is not None:
                omega = pure_pursuit_omega(v, float(goal[0]), float(goal[1]))
                self._publish_lookahead(goal)

        if omega is None:
            if use_pure_pursuit:
                self.get_logger().warn(
                    "Pure Pursuit 用の経路が足りないので生の dyaw にフォールバックします",
                    throttle_duration_sec=5.0,
                )
            # 従来方式: 選んだ行動の dyaw をそのまま角速度にする。
            omega = float(selected[1]) / DT if selected is not None else 0.0

        omega = float(np.clip(omega, -self.angular_max_vel, self.angular_max_vel))
        cmd_vel = Twist()
        cmd_vel.linear.x = v
        cmd_vel.angular.z = omega
        self.cmd_vel_pub.publish(cmd_vel)

    def _publish_lookahead(self, goal: np.ndarray) -> None:
        """Pure Pursuit の前方注視点を publish する（注視距離の調整用）。"""
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.path_frame_id
        msg.point.x = float(goal[0])
        msg.point.y = float(goal[1])
        self.lookahead_pub.publish(msg)

    def _history_frames(self) -> Optional[list[np.ndarray]]:
        """履歴バッファから [現在, 1秒前, 2秒前] の3枚を取り出す。

        まだ 11 枚溜まっていない（起動直後・自律ON直後）場合はインデックスを 0 に
        クランプする＝バッファ内で最も古いフレームを複製して padding する。学習側の
        「エピソード先頭は frame 0 を複製」と同じ意味になる（決定4）。
        履歴が空なら None を返す（呼び出し側で推論をスキップ）。
        """
        with self._history_lock:
            hist = list(self.image_history)
        if not hist:
            return None

        def pick(lag: int) -> np.ndarray:
            return hist[max(len(hist) - 1 - lag, 0)]

        return [pick(0), pick(HISTORY_STRIDE_FRAMES), pick(2 * HISTORY_STRIDE_FRAMES)]

    def _publish_pred_path(self, chunk: np.ndarray) -> None:
        """直近に推論したchunk（今後10秒分の予測）をそのまま base_link 基準の Path として publish。

        過去は蓄積しない。新しいchunkが来るたびに置き換え（毎回 poses を作り直す）。
        base_link 基準 = 常に「現在のロボット位置」からの相対軌跡として解釈される。
        """
        stamp = self.get_clock().now().to_msg()
        x = y = theta = 0.0
        poses: list[PoseStamped] = []
        for dx_body, dyaw in chunk:
            x += float(dx_body) * math.cos(theta)
            y += float(dx_body) * math.sin(theta)
            theta += float(dyaw)

            pose = PoseStamped()
            pose.header.stamp = stamp
            pose.header.frame_id = self.path_frame_id
            pose.pose.position.x = x
            pose.pose.position.y = y
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
        """残量に関係なく、最新観測で毎回 chunk を計算し、
        絶対時刻でそろえて既存キューに集約する（lerobot _aggregate_action_queues 相当）。

        以前は g=0.7（30%消費で再推論、SmolVLA論文 3.3節）の閾値で間引いていたが、
        常に推論し続けて行動を更新し続けるようにするため閾値を撤廃した。
        MutuallyExclusiveCallbackGroup なので ~1.2s ブロックしても制御タイマーは
        別スレッドで回り続け、推論の二重起動もしない（前回の推論が終わり次第、即座に次の推論が始まる）。
        """
        if not self.autonomous_flag or self.latest_image is None:
            return

        with self._queue_lock:
            base_step = self._step   # この観測が予測する行動列の起点となる絶対時刻

        # 履歴バッファから時系列3枚を取り出す（揃っていなければ最古フレームで複製）。
        images = self._history_frames()
        if images is None:
            return   # 制御ループがまだ1枚も push していない（自律ON直後）

        # 参照代入は GIL 下で原子的なので、最新値をスナップショットして使う。
        prompt = self.latest_prompt
        # state = [v, omega]。暫定ゼロ固定（copycat 対策で現在指令は入れない）。
        state = np.zeros(2, dtype=np.float32)

        # ここが重い（~1.2s）。制御ループとは別スレッドなので停止しない。
        # chunk[i] は絶対時刻 base_step + i の行動に対応する。
        chunk = self.model.infer_chunk(images, state, prompt)  # (chunk_size, 2)

        # 今回のchunk(=今後10秒の予測、クリップ前の生値)をそのままRViz可視化用に publish。
        self._publish_pred_path(chunk)

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
