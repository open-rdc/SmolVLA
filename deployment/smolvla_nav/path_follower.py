#!/usr/bin/env python3
"""/smolvla_pred_path を Pure Pursuit で追従し、最終的な /cmd_vel を発行するノード。

navigation.py は SmolVLA の推論・並進速度(v)の計算・/smolvla_pred_path の publish だけを
行い、最終的な /cmd_vel は publish しない（/smolvla_cmd_vel_raw に v とフォールバック用の
生 dyaw だけを出す）。操舵の決定はこのノードに分離してある。

分離した経緯: navigation.py に元々あった Pure Pursuit 実装は
`pure_pursuit_omega(v, goal[10], goal[11])` のように前方注視点(2要素配列)を誤った
インデックスで参照しており、毎 tick IndexError で例外を起こして cmd_vel の publish
まで到達しなかった（＝最後にpublishされた「直進」指令が固まって出続けていた）。
Pure Pursuit の計算・publish 経路を独立したノードに切り出すことで、単体でも
デバッグ・テストしやすくしている。

購読:
  /smolvla_pred_path    (nav_msgs/Path)     navigation.py が publish する予測経路
                                              (base_link 基準、直近の推論結果そのもの)
  /smolvla_cmd_vel_raw  (geometry_msgs/Twist) navigation.py が publish する v とフォールバック omega
  /autonomous           (std_msgs/Bool)      自律走行フラグ

publish:
  /cmd_vel              (geometry_msgs/Twist) 実際にロボットへ渡す最終指令
  /smolvla_lookahead    (geometry_msgs/PointStamped) Pure Pursuit の前方注視点(可視化用)

既知の制約: /smolvla_pred_path は推論のたびに(~1.2s間隔)更新される、その時点の
base_link を基準にしたスナップショット。次の推論が来るまで同じ経路を使い続けるため、
その間にロボットが動いた分だけ注視点が実際の相対位置からずれる。経路追従の粗い
近似としては十分だが、精度が要る場合は推論頻度を上げるか、navigation.py 側の
絶対時刻キュー(_actions)を都度再基準化して露出する経路にする改良が必要。
"""

from __future__ import annotations

from typing import Optional

import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, Twist
from nav_msgs.msg import Path as NavPath
from std_msgs.msg import Bool

from smolvla_nav.pure_pursuit import find_lookahead_point, pure_pursuit_omega

CONTROL_INTERVAL_MS = 100   # /smolvla_cmd_vel_raw(200ms周期)より少し細かく回して追従を滑らかにする

DEFAULT_USE_PURE_PURSUIT = True
DEFAULT_LOOKAHEAD_DISTANCE = 1.0   # 前方注視距離[m]。不感帯(約1〜2m)より長く取ること
DEFAULT_ANGULAR_MAX_VEL = 1.0
# /smolvla_pred_path がこれより古ければ経路なし扱い。
# 実測(RTX 2070 Max-Q + SmolVLM2-500M): infer_chunk は定常状態で約2.6〜3.5秒/回、
# 初回はモデルのウォームアップで約6.9秒かかる（navigation_node の [latency] ログ参照）。
# ドキュメント中に残っていた「推論は~1.2s」という想定は実測より大幅に楽観的だった。
# 2.0s のままだと定常状態でも毎回タイムアウトしてPure Pursuitが実質使われず、常に
# 生dyawへフォールバックしていた。定常latencyに余裕を持たせて 5.0s にする。
DEFAULT_PATH_TIMEOUT_SEC = 5.0
DEFAULT_RAW_TIMEOUT_SEC = 1.0      # /smolvla_cmd_vel_raw がこれより古ければ安全停止


def path_to_xy(path_msg: NavPath) -> np.ndarray:
    """nav_msgs/Path から (N, 2) の [x, y] 配列を取り出す（base_link 基準はそのまま）。"""
    return np.asarray(
        [[p.pose.position.x, p.pose.position.y] for p in path_msg.poses],
        dtype=np.float64,
    )


class PathFollowerNode(Node):
    def __init__(self) -> None:
        super().__init__("smolvla_path_follower")

        self.declare_parameter("use_pure_pursuit", DEFAULT_USE_PURE_PURSUIT)
        self.declare_parameter("lookahead_distance", DEFAULT_LOOKAHEAD_DISTANCE)
        self.declare_parameter("angular_max_vel", DEFAULT_ANGULAR_MAX_VEL)
        self.declare_parameter("path_timeout_sec", DEFAULT_PATH_TIMEOUT_SEC)
        self.declare_parameter("raw_timeout_sec", DEFAULT_RAW_TIMEOUT_SEC)

        self.autonomous_flag = False

        self.latest_path_xy: Optional[np.ndarray] = None
        self.path_stamp: Optional[rclpy.time.Time] = None
        self.path_frame_id = "base_link"

        self.latest_raw: Optional[Twist] = None
        self.raw_stamp: Optional[rclpy.time.Time] = None

        self.path_sub = self.create_subscription(NavPath, "/smolvla_pred_path", self.path_callback, 10)
        self.raw_cmd_vel_sub = self.create_subscription(
            Twist, "/smolvla_cmd_vel_raw", self.raw_cmd_vel_callback, 10
        )
        self.autonomous_sub = self.create_subscription(Bool, "/autonomous", self.autonomous_callback, 10)

        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        # 注視点を /smolvla_pred_path と重ねて見ると、注視点が計画の「まっすぐな
        # 前置き（不感帯）」の中に入っていないか（=注視距離が短すぎないか）を目視できる。
        self.lookahead_pub = self.create_publisher(PointStamped, "/smolvla_lookahead", 10)

        self.control_timer = self.create_timer(CONTROL_INTERVAL_MS / 1000.0, self.control_timer_callback)

    # ---- callbacks ----------------------------------------------------
    def autonomous_callback(self, msg: Bool) -> None:
        self.autonomous_flag = msg.data

    def path_callback(self, msg: NavPath) -> None:
        if msg.poses:
            self.latest_path_xy = path_to_xy(msg)
            self.path_frame_id = msg.header.frame_id or self.path_frame_id
        else:
            self.latest_path_xy = None
        self.path_stamp = self.get_clock().now()

    def raw_cmd_vel_callback(self, msg: Twist) -> None:
        self.latest_raw = msg
        self.raw_stamp = self.get_clock().now()

    # ---- 制御ループ -----------------------------------------------------
    def control_timer_callback(self) -> None:
        if not self.autonomous_flag:
            return  # 非自律時は publish しない（他コントローラに任せる）

        raw_timeout_sec = float(self.get_parameter("raw_timeout_sec").value)
        if self.latest_raw is None or self.raw_stamp is None or self._age(self.raw_stamp) > raw_timeout_sec:
            # v の元になる /smolvla_cmd_vel_raw が無い/古い -> 安全のため停止指令。
            self.cmd_vel_pub.publish(Twist())
            return

        v = float(self.latest_raw.linear.x)
        fallback_omega = float(self.latest_raw.angular.z)

        use_pure_pursuit = bool(self.get_parameter("use_pure_pursuit").value)
        lookahead_distance = float(self.get_parameter("lookahead_distance").value)
        angular_max_vel = float(self.get_parameter("angular_max_vel").value)
        path_timeout_sec = float(self.get_parameter("path_timeout_sec").value)

        omega = fallback_omega
        if use_pure_pursuit:
            if self.latest_path_xy is None or self.path_stamp is None:
                self.get_logger().warn(
                    "/smolvla_pred_path 未受信のため生の dyaw にフォールバックします",
                    throttle_duration_sec=5.0,
                )
            elif len(self.latest_path_xy) < 2:
                self.get_logger().warn(
                    f"/smolvla_pred_path の点数が足りない(len={len(self.latest_path_xy)})"
                    "ため生の dyaw にフォールバックします",
                    throttle_duration_sec=5.0,
                )
            elif (age := self._age(self.path_stamp)) > path_timeout_sec:
                self.get_logger().warn(
                    f"/smolvla_pred_path が古い(age={age:.2f}s > timeout={path_timeout_sec:.2f}s)"
                    "ため生の dyaw にフォールバックします。推論レイテンシが timeout を"
                    "超えている可能性があるので navigation_node の [latency] ログを確認してください。",
                    throttle_duration_sec=5.0,
                )
            else:
                goal = find_lookahead_point(self._as_poses(self.latest_path_xy), lookahead_distance)
                if goal is not None:
                    omega = pure_pursuit_omega(v, float(goal[0]), float(goal[1]))
                    self._publish_lookahead(goal)

        omega = float(np.clip(omega, -angular_max_vel, angular_max_vel))

        cmd_vel = Twist()
        cmd_vel.linear.x = v
        cmd_vel.angular.z = omega
        self.cmd_vel_pub.publish(cmd_vel)

    def _age(self, stamp: rclpy.time.Time) -> float:
        return (self.get_clock().now() - stamp).nanoseconds / 1e9

    @staticmethod
    def _as_poses(xy: np.ndarray) -> np.ndarray:
        """find_lookahead_point が期待する (N, 3) 形式に詰め替える（theta は未使用なので0埋め）。"""
        poses = np.zeros((len(xy), 3), dtype=np.float64)
        poses[:, :2] = xy
        return poses

    def _publish_lookahead(self, goal: np.ndarray) -> None:
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.path_frame_id
        msg.point.x = float(goal[0])
        msg.point.y = float(goal[1])
        self.lookahead_pub.publish(msg)


def main() -> int:
    rclpy.init()
    node = PathFollowerNode()
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
