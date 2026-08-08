#!/usr/bin/env python3
"""path_follower.py の control_timer_callback を、ROSをスタブして実際に動かすテスト。

pure_pursuit.py は本物をそのまま使う（ROS非依存なので軽い）。ここでは
/smolvla_pred_path -> Pure Pursuit -> /cmd_vel の分岐ロジック（不感帯の飛び越え、
経路なし/古い/raw古い時のフォールバック・安全停止、クリップ）を検証する。
"""
import sys
from pathlib import Path
import types
from unittest import mock

import numpy as np

# ---- 重い依存をスタブしてから path_follower を import する ----------------
# (smolvla_nav.pure_pursuit は numpy だけに依存する軽量モジュールなので本物を使う)
class _Msg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class Vec:
    def __init__(self):
        self.x = self.y = self.z = 0.0


class Twist:
    def __init__(self):
        self.linear = Vec()
        self.angular = Vec()


class _Hdr:
    def __init__(self):
        self.stamp = None
        self.frame_id = ""


class PointStamped:
    def __init__(self):
        self.header = _Hdr()
        self.point = Vec()


for name in ["rclpy", "rclpy.node", "geometry_msgs", "geometry_msgs.msg",
             "nav_msgs", "nav_msgs.msg", "std_msgs", "std_msgs.msg"]:
    sys.modules.setdefault(name, types.ModuleType(name))

sys.modules["rclpy.node"].Node = type("Node", (), {})
sys.modules["geometry_msgs.msg"].Twist = Twist
sys.modules["geometry_msgs.msg"].PointStamped = PointStamped
sys.modules["nav_msgs.msg"].Path = _Msg
sys.modules["std_msgs.msg"].Bool = _Msg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from smolvla_nav import path_follower as pf  # noqa: E402
from smolvla_nav.pure_pursuit import integrate_path  # noqa: E402

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")
    if not cond:
        fails.append(name)


class StubTime:
    """rclpy.time.Time の代わり。ナノ秒だけを保持して引き算できればよい。"""

    def __init__(self, ns):
        self.ns = ns

    def __sub__(self, other):
        return types.SimpleNamespace(nanoseconds=self.ns - other.ns)

    def to_msg(self):
        return None


def make_node(path_xy=None, raw=None, params=None, now_ns=0, path_age_sec=0.0, raw_age_sec=0.0):
    n = object.__new__(pf.PathFollowerNode)
    n.autonomous_flag = True
    n.path_frame_id = "base_link"
    n.latest_path_xy = path_xy
    n.path_stamp = StubTime(now_ns - int(path_age_sec * 1e9)) if path_xy is not None else None
    n.latest_raw = raw
    n.raw_stamp = StubTime(now_ns - int(raw_age_sec * 1e9)) if raw is not None else None

    default_params = {
        "use_pure_pursuit": True,
        "lookahead_distance": pf.DEFAULT_LOOKAHEAD_DISTANCE,
        "angular_max_vel": pf.DEFAULT_ANGULAR_MAX_VEL,
        "path_timeout_sec": pf.DEFAULT_PATH_TIMEOUT_SEC,
        "raw_timeout_sec": pf.DEFAULT_RAW_TIMEOUT_SEC,
    }
    if params:
        default_params.update(params)
    n.get_parameter = lambda name: types.SimpleNamespace(value=default_params[name])
    n.cmd_vel_pub = mock.Mock()
    n.lookahead_pub = mock.Mock()
    n.get_logger = lambda: mock.Mock()
    n.get_clock = lambda: types.SimpleNamespace(now=lambda: StubTime(now_ns))
    return n


def run(n):
    n.control_timer_callback()
    assert n.cmd_vel_pub.publish.called, "cmd_vel が publish されていない"
    return n.cmd_vel_pub.publish.call_args[0][0]


def make_raw(v, fallback_omega):
    t = Twist()
    t.linear.x = v
    t.angular.z = fallback_omega
    return t


DX = 0.195
DT = 0.2
V = DX / DT

print("=== 1. 非自律なら何も publish しない ===")
n = make_node(raw=make_raw(V, 0.0))
n.autonomous_flag = False
n.control_timer_callback()
check("publish されない", not n.cmd_vel_pub.publish.called)

print("=== 2. raw が届いていない -> 安全停止 ===")
n = make_node(raw=None)
t = run(n)
check("v=0", t.linear.x == 0.0)
check("omega=0", t.angular.z == 0.0)

print("=== 3. raw が古すぎる -> 安全停止 ===")
n = make_node(raw=make_raw(V, 0.02), raw_age_sec=5.0)
t = run(n)
check("v=0", t.linear.x == 0.0)
check("omega=0", t.angular.z == 0.0)

print("=== 4. Pure Pursuit OFF -> raw の angular.z をそのまま使う ===")
n = make_node(raw=make_raw(V, 0.02), params={"use_pure_pursuit": False})
t = run(n)
check("v はrawそのまま", abs(t.linear.x - V) < 1e-6)
check("omega はrawそのまま", abs(t.angular.z - 0.02) < 1e-6)

print("=== 5. Pure Pursuit が不感帯を飛び越える（本題）===")
# 実機で観測された形: 1.4秒(7step)まっすぐ -> その後ゆるやかに左へ戻る
plan = [np.array([DX, 0.0])] * 7 + [np.array([DX, 0.012])] * 43
path_xy = integrate_path(np.asarray(plan))[:, :2]

t_short = run(make_node(path_xy=path_xy, raw=make_raw(V, 0.0), params={"lookahead_distance": 1.0}))
t_long = run(make_node(path_xy=path_xy, raw=make_raw(V, 0.0), params={"lookahead_distance": 2.5}))
print(f"     L=1.0m(不感帯内) omega={t_short.angular.z:.4f}")
print(f"     L=2.5m(不感帯外) omega={t_long.angular.z:.4f}")
check("L=1.0m でも omega≈0（注視距離が短すぎると効かない）", abs(t_short.angular.z) < 1e-6)
check("L=2.5m で omega>0（不感帯を飛び越える）", t_long.angular.z > 0.005)
check("v は PP でも変わらない（rawのvをそのまま使う）", abs(t_long.linear.x - V) < 1e-6)

print("=== 6. PP ON で注視点が publish される ===")
n = make_node(path_xy=path_xy, raw=make_raw(V, 0.0), params={"lookahead_distance": 2.5})
run(n)
check("lookahead が publish される", n.lookahead_pub.publish.called)
pt = n.lookahead_pub.publish.call_args[0][0]
check("注視点は 2.5m 以上先", np.hypot(pt.point.x, pt.point.y) >= 2.5,
      f"({pt.point.x:.2f}, {pt.point.y:.2f})")

print("=== 7. PP ON だが経路が短すぎる -> 生の dyaw にフォールバック ===")
t = run(make_node(path_xy=np.array([[0.1, 0.0]]), raw=make_raw(V, 0.04)))
check("raw の omega を使う", abs(t.angular.z - 0.04) < 1e-6, f"omega={t.angular.z:.4f}")

print("=== 8. PP ON だが経路が古い -> 生の dyaw にフォールバック ===")
t = run(make_node(path_xy=path_xy, raw=make_raw(V, 0.04), path_age_sec=10.0))
check("raw の omega を使う", abs(t.angular.z - 0.04) < 1e-6, f"omega={t.angular.z:.4f}")

print("=== 9. クリップ ===")
n = make_node(path_xy=path_xy, raw=make_raw(V, 0.0), params={"lookahead_distance": 0.01, "angular_max_vel": 0.05})
t = run(n)
check("omega が上限でクリップ", t.angular.z <= 0.05 + 1e-9, f"omega={t.angular.z}")

print()
print("FAILED:", fails if fails else "なし")
sys.exit(1 if fails else 0)
