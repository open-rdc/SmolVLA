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
import os
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

# --- 経路追従モードの既定値（既定は OFF = 従来どおり生の dyaw をそのまま流す）---
# 実機で「コース端に寄ると中央に戻れない」問題の切り分け/対策用。モデルの計画は
# 「1〜2秒まっすぐ → その後ゆるやかに戻る」形をしているが、推論が ~1.2s かかるため
# chunk の先頭6ステップ程度しか実行されず、曲がり始める部分に到達しない。
# 実験中に ros2 param set で切り替えたいので ROS2 パラメータにしてある。
# 注意: Pure Pursuit による操舵の置き換えは path_follower.py（別ノード）が
# /smolvla_pred_path を購読して行う。ここでは publish しない（決定: pred_path が
# 正しく cmd_vel に反映されていなかったバグ修正で分離した）。
DEFAULT_STEP_LOOKAHEAD = 0          # chunk の何ステップ先の行動を使うか（不感帯の実測用）

# デノイズのステップ数。0 ならポリシー側の既定（num_steps=4）に従う。
#
# 積分は Heun法（2段2次）で固定されており、1ステップにつき速度場を2回評価する。
# 推論時間は「ステップ数」ではなく「評価回数 = 2 × num_steps」に比例するので、
# 既定の 4 は評価8回。以前の陽的オイラー法 num_steps=10（評価10回）より速く、
# 実測では誤差も約半分になる。
#
# ⚠ 2 以下に下げると、ステップ幅が大きすぎて修正子が割に合わない領域に入る
#   （実測では評価4回で互角、2回では1次のオイラー法のほうが正確）。
#   オイラー法に戻したい場合は feat/heun-solver 以前のブランチを使うこと。
DEFAULT_NUM_STEPS = 4               # 0 なら上書きしない。チェックポイントのconfig.jsonの値(10)より優先

# colcon install 後は __file__ が site-packages 配下になり parents[2] ではリポジトリルートに
# 届かないため、env_humble.sh が設定する SMOLVLA_REPO_ROOT を優先する。未設定時（ソースツリーから
# 直接実行する場合）だけ従来通り __file__ から逆算する。
_env_repo_root = os.environ.get("SMOLVLA_REPO_ROOT")
_REPO_ROOT = Path(_env_repo_root) if _env_repo_root else Path(__file__).resolve().parents[2]

# チェックポイントの場所（tar.gz を展開した先）
DEFAULT_CKPT = _REPO_ROOT / "training" / "data" / "weight" / "smolvla_orne_tc_ms_rec5_ckpt" / "pretrained_model"


# ══════════════════════════════════════════════════════════════════
#  SmolVLA ラッパー
# ══════════════════════════════════════════════════════════════════
class SmolVLAModel:
    """学習済み SmolVLA をロードし、1 枚の画像+状態+指示から action を返す。"""

    def __init__(
        self,
        ckpt_dir: Path = DEFAULT_CKPT,
        device: Optional[str] = None,
        num_steps: int = DEFAULT_NUM_STEPS,
    ) -> None:
        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))

        # 1) ポリシー本体（重み込み）をロード。config.json / model.safetensors を読む。
        self.policy = SmolVLAPolicy.from_pretrained(str(ckpt_dir))

        # 2) デノイズのステップ数の上書き。推論時にしか効かないので重みは取り直さない。
        #    チェックポイントの config.json に num_steps=10 が保存されていても、
        #    ここで上書きすれば新しい既定を使える。
        if num_steps > 0:
            self.policy.config.num_steps = num_steps
        self.num_steps = self.policy.config.num_steps
        # 推論時間はステップ数ではなく速度場の評価回数に比例する。Heun は1ステップ2評価。
        self.nfe = self.num_steps * 2

        self.policy.to(self.device).eval()

        # vision encoder(SigLIP)を torch.compile でカーネル融合させる案は無効化。
        # RTX 2070 Max-Q だと "Not enough SMs to use max_autotune_gemm mode" /
        # "does not support bfloat16 compilation natively" という警告が出るほど
        # 相性が悪く、コンパイルが非常に長時間かかる(またはハングする)ため
        # /smolvla_pred_path が全く出ない状態になった。効果も575ms→450ms程度と
        # 大きくなかったため、いったん無効化して切り分ける。

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

        # フレームid -> 画像埋め込みtensor のキャッシュ。時系列3枚(camera1/2/3)は履歴バッファの
        # スライドにより後続の呼び出しで同じ物理フレームを指すことが多いため、vision encoderの
        # 再計算を避けられる。呼び出し側(SmolVLANavigationNode)がフレームごとに安定したidを
        # 割り当てて infer_chunk() に渡す。
        self._image_embed_cache: dict[int, torch.Tensor] = {}

    def reset(self) -> None:
        """自律走行を開始/再開するたびに呼ぶ（内部の action chunk キューを空にする）。"""
        self.policy.reset()
        self._image_embed_cache.clear()

    def prune_image_embed_cache(self, valid_ids: set[int]) -> None:
        """もう履歴バッファに残っていないフレームのキャッシュを捨てる（無限に増えないように）。"""
        stale = [k for k in self._image_embed_cache if k not in valid_ids]
        for k in stale:
            del self._image_embed_cache[k]

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
    def infer_chunk(
        self,
        images_rgb: list[np.ndarray],
        state: np.ndarray,
        task: str,
        image_ids: Optional[list[int]] = None,
    ) -> np.ndarray:
        """3枚の時系列観測から 50 ステップ分の行動列をまとめて返す（非同期先読み用）。

        select_action は内部キューから1手ずつ返すが、こちらは chunk 全体を返すので
        呼び出し側で自前キューを管理できる。

        Args:
            images_rgb: [現在(t), 1つ前(t-1), 2つ前(t-2)] の3枚。infer() と同じ形式。
            image_ids: images_rgb の各フレームに対応する安定したid（省略時はキャッシュ無効）。
                       同じidが渡された画像は前回計算した埋め込みを再利用する。

        Returns:
            actions: np.ndarray shape (chunk_size, 2) = [[dx_body, dyaw], ...]
        """
        batch = self._build_batch(images_rgb, state, task)
        batch = self.preprocessor(batch)
        kwargs = {"image_ids": image_ids, "image_embed_cache": self._image_embed_cache}
        if self.device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                chunk = self.policy.predict_action_chunk(batch, **kwargs)  # (1, chunk_size, action_dim)
        else:
            chunk = self.policy.predict_action_chunk(batch, **kwargs)
        chunk = self.postprocessor(chunk)
        return chunk.squeeze(0).float().cpu().numpy()  # (chunk_size, 2)


# ══════════════════════════════════════════════════════════════════
#  ROS2 ノード
# ══════════════════════════════════════════════════════════════════


class SmolVLANavigationNode(Node):
    def __init__(self) -> None:
        super().__init__("smolvla_navigation")

        # --- モデル（実装済み）---
        # ステップ数は起動時にだけ効く（推論スレッドが走り出す前にモデルを作るため）。
        # 変えたい場合はノードを立て直すこと。
        self.declare_parameter("num_steps", DEFAULT_NUM_STEPS)
        self.model = SmolVLAModel(num_steps=int(self.get_parameter("num_steps").value))
        self.get_logger().info(
            f"SmolVLA loaded. ODE=Heun法(2段2次) num_steps={self.model.num_steps} "
            f"(速度場の評価 {self.model.nfe} 回/推論)"
        )

        # --- 状態変数 ---
        self.autonomous_flag = False
        self.latest_image: Optional[np.ndarray] = None   # RGB, (IMG_H, IMG_W, 3)
        self.latest_prompt = "go straight along the road"
        self.state = np.zeros(2, dtype=np.float32)        # [v, omega]

        # --- 時系列コンテキストフレーム ---
        # 元は学習データのフレーム間隔(dt=0.2s)に厳密に揃えて「1秒前」「2秒前」を
        # tick単位で取り出していた（決定3、smolvla-temporal-context-architecture.md）。
        # だが推論レイテンシ(650〜950ms)が1秒未満になったことで、tick基準の1秒前/2秒前
        # フレームは呼び出しのたびに新規フレームとなり、画像埋め込みキャッシュが
        # 実質ヒットしなかった（詳細はコミット時のやり取り参照）。
        # ここでは「1秒前/2秒前」を実時間ではなく「前回/前々回の推論で"今"として使った
        # フレーム」と定義し直す。こうすると camera2/camera3 は必ず既に埋め込み済みに
        # なりキャッシュが確実にヒットする代わりに、実際の時間間隔は推論レイテンシに
        # 依存して変動する(厳密に1秒/2秒ではない)。レイテンシ計測用途では許容できる
        # 近似だが、実走行の精度に使う場合は実データで要検証。
        self._context_frames: collections.deque[tuple[int, np.ndarray]] = collections.deque(maxlen=2)
        self._frame_counter = 0

        # --- パラメータ（速度上限・制御周期）---
        self.linear_max_vel = 1.0
        self.angular_max_vel = 1.0
        self.interval_ms = 200                 # 制御周期 = DT(200ms) と揃える

        # --- 経路追従モード（復帰動作の実験用。既定は OFF = 従来の挙動）---
        # step_lookahead: chunk の K ステップ先の行動を使う。計画の先頭にある
        #   「まっすぐな前置き（不感帯）」を飛ばせるので、K を 0→5→10→15 と振ると
        #   不感帯の長さがステップ数で実測できる。0 = 従来どおり現在stepの行動。
        # Pure Pursuit による操舵の置き換えは path_follower.py が担当する。
        self.declare_parameter("step_lookahead", DEFAULT_STEP_LOOKAHEAD)

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
        # 最終的な /cmd_vel は path_follower.py が publish する。ここでは v(linear.x)と
        # フォールバック用の生 dyaw(angular.z) だけを "raw" として出す。
        self.cmd_vel_pub = self.create_publisher(Twist, "/smolvla_cmd_vel_raw", 10)
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
        # 自律 OFF -> ON の立ち上がりで行動キューをリセット（前エピソードの残りを捨てる）。
        if msg.data and not self.autonomous_flag:
            self.model.reset()
            with self._queue_lock:
                self._actions.clear()
                self._step = 0
            # 前回走行のコンテキストフレームを持ち越さない（行動キューのクリアと同じ理由）。
            self._context_frames.clear()
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
        """200ms ごとに現在 step の行動を取り出して /smolvla_cmd_vel_raw を発行する。推論はしない。

        ここで出す angular.z は「生の dyaw（フォールバック値）」でしかない。実際に
        走行で使う最終的な /cmd_vel の操舵は、/smolvla_pred_path を見て Pure Pursuit
        で計算する path_follower.py（別ノード）が担当する。
        """
        if not self.autonomous_flag:
            return  # 非自律時は publish しない（他コントローラに任せる）

        # 走行中に ros2 param set で切り替えられるよう毎tick読み直す（5Hzなので負荷は無視できる）。
        step_lookahead = max(int(self.get_parameter("step_lookahead").value), 0)

        # 現在 step の行動を取り出し、時刻を1つ進める（行動が無くても step は進める）。
        with self._queue_lock:
            step = self._step
            current = self._actions.get(step)      # 「今」の行動（並進速度はこれを使う）
            selected = current                     # フォールバック操舵に使う行動
            if step_lookahead:
                # K ステップ先の行動を使う。まだ届いていなければ現在stepにフォールバック。
                ahead = self._actions.get(step + step_lookahead)
                if ahead is not None:
                    selected = ahead
            self._actions.pop(step, None)          # 消化（辞書が無限に伸びないように）
            self._step += 1

        if current is None and selected is None:
            # まだ chunk が用意できていない（起動直後など）→ 安全のため停止指令。
            self.cmd_vel_pub.publish(Twist())
            return

        speed_src = current if current is not None else selected
        v = float(np.clip(float(speed_src[0]) / DT, -self.linear_max_vel, self.linear_max_vel))
        omega = float(selected[1]) / DT if selected is not None else 0.0
        omega = float(np.clip(omega, -self.angular_max_vel, self.angular_max_vel))

        cmd_vel = Twist()
        cmd_vel.linear.x = v
        cmd_vel.angular.z = omega
        self.cmd_vel_pub.publish(cmd_vel)

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

        # 「今」は常に最新フレームを新規エンコードする。「1秒前」「2秒前」は実時間ではなく
        # 前回・前々回の推論で"今"として使ったフレームを流用する（キャッシュ確実ヒット）。
        # まだ2回に満たない起動直後は、無い分を現在フレームで埋める。
        current_image = self.latest_image     # 参照代入は GIL 下で原子的
        self._frame_counter += 1
        current_id = self._frame_counter

        context = list(self._context_frames)   # [1回前, 2回前]（無ければ短い）
        cam2_id, cam2_img = context[-1] if len(context) >= 1 else (current_id, current_image)
        cam3_id, cam3_img = context[-2] if len(context) >= 2 else (cam2_id, cam2_img)

        image_ids = [current_id, cam2_id, cam3_id]
        images = [current_image, cam2_img, cam3_img]

        self._context_frames.append((current_id, current_image))

        # もうcamera2/camera3として参照されえないフレームの画像埋め込みキャッシュを捨てる。
        valid_ids = {current_id, cam2_id, cam3_id} | {fid for fid, _ in self._context_frames}
        self.model.prune_image_embed_cache(valid_ids)

        # 参照代入は GIL 下で原子的なので、最新値をスナップショットして使う。
        prompt = self.latest_prompt
        # state = [v, omega]。暫定ゼロ固定（copycat 対策で現在指令は入れない）。
        state = np.zeros(2, dtype=np.float32)

        # ここが重い（~1.2s想定）。制御ループとは別スレッドなので停止しない。
        # chunk[i] は絶対時刻 base_step + i の行動に対応する。
        # レイテンシを計測して毎回ログに出す（path_follower の path_timeout_sec を
        # 実測値に合わせて調整するための材料。5Hzで回るのでスパム防止に1秒間隔で間引く）。
        infer_start = self.get_clock().now()
        chunk = self.model.infer_chunk(images, state, prompt, image_ids=image_ids)  # (chunk_size, 2)
        latency_sec = (self.get_clock().now() - infer_start).nanoseconds / 1e9
        # VRAM も併せて出す。allocated=実際に使用中 / reserved=PyTorchがOSから確保済み。
        # 両方増える            -> リーク
        # allocated は平坦で reserved だけ増える -> アロケータの断片化
        # どちらも平坦なのに latency だけ伸びる   -> GPU外（電力制限・熱・CPU競合）
        if self.model.device.type == "cuda":
            vram = (
                f" | VRAM alloc {torch.cuda.memory_allocated() / 2**20:.0f} MiB"
                f" / reserved {torch.cuda.memory_reserved() / 2**20:.0f} MiB"
            )
        else:
            vram = ""
        self.get_logger().info(
            f"[latency] infer_chunk: {latency_sec * 1000:.0f} ms{vram}", throttle_duration_sec=1.0
        )

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
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
