#!/usr/bin/env python3
"""navigation.py の control_timer_callback を、ROS/torch をスタブして実際に動かすテスト。

実物のメソッドを呼ぶので、分岐ロジック（+K のフォールバック、Pure Pursuit、
停止指令、辞書のGC、クリップ）がそのまま検証される。
"""
import sys
from pathlib import Path
import threading
import types
from collections import deque
from unittest import mock

import numpy as np

# ---- 重い依存をスタブしてから navigation を import する ----------------
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


for name in ["cv2", "torch", "lerobot", "lerobot.policies", "lerobot.policies.factory",
             "lerobot.policies.smolvla", "lerobot.policies.smolvla.modeling_smolvla",
             "rclpy", "rclpy.node", "rclpy.callback_groups", "rclpy.executors",
             "geometry_msgs", "geometry_msgs.msg", "nav_msgs", "nav_msgs.msg",
             "sensor_msgs", "sensor_msgs.msg", "std_msgs", "std_msgs.msg",
             "smolvla_nav.image_convert"]:
    sys.modules.setdefault(name, types.ModuleType(name))

sys.modules["torch"].no_grad = lambda: (lambda f: f)   # @torch.no_grad() はクラス定義時に評価される
sys.modules["rclpy.node"].Node = type("Node", (), {})
sys.modules["rclpy.callback_groups"].MutuallyExclusiveCallbackGroup = object
sys.modules["rclpy.executors"].MultiThreadedExecutor = object
sys.modules["geometry_msgs.msg"].Twist = Twist
sys.modules["geometry_msgs.msg"].PoseStamped = _Msg
sys.modules["geometry_msgs.msg"].PointStamped = PointStamped
sys.modules["nav_msgs.msg"].Path = _Msg
sys.modules["sensor_msgs.msg"].Image = _Msg
sys.modules["std_msgs.msg"].Bool = _Msg
sys.modules["std_msgs.msg"].String = _Msg
sys.modules["lerobot.policies.factory"].make_pre_post_processors = None
sys.modules["lerobot.policies.smolvla.modeling_smolvla"].SmolVLAPolicy = None
sys.modules["smolvla_nav.image_convert"].image_msg_to_bgr = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from smolvla_nav import navigation as nav  # noqa: E402

DT = nav.DT
fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")
    if not cond:
        fails.append(name)


def make_node(actions, step=0, k=0, pp=False, L=2.5):
    """control_timer_callback を呼べる最小のスタブ node を作る。"""
    n = object.__new__(nav.SmolVLANavigationNode)
    n.autonomous_flag = True
    n.latest_image = None
    n.image_history = deque(maxlen=nav.HISTORY_LEN)
    n._history_lock = threading.Lock()
    n._queue_lock = threading.Lock()
    n._step = step
    n._actions = dict(actions)
    n.linear_max_vel = 1.0
    n.angular_max_vel = 1.0
    n.path_frame_id = "base_link"
    params = {"step_lookahead": k, "use_pure_pursuit": pp, "lookahead_distance": L}
    n.get_parameter = lambda name: types.SimpleNamespace(value=params[name])
    n.cmd_vel_pub = mock.Mock()
    n.lookahead_pub = mock.Mock()
    n.get_logger = lambda: mock.Mock()
    n.get_clock = lambda: types.SimpleNamespace(now=lambda: types.SimpleNamespace(to_msg=lambda: None))
    return n


def run(n):
    n.control_timer_callback()
    assert n.cmd_vel_pub.publish.called, "cmd_vel が publish されていない"
    return n.cmd_vel_pub.publish.call_args[0][0]


DX = 0.195
V = DX / DT                       # 約 0.975 m/s

print("=== 1. キューが空 -> 停止指令 ===")
t = run(make_node({}))
check("v=0", t.linear.x == 0.0)
check("omega=0", t.angular.z == 0.0)

print("=== 2. K=0, PP off -> 従来どおり生の dyaw ===")
acts = {i: np.array([DX, 0.02], np.float32) for i in range(20)}
t = run(make_node(acts))
check("v = dx/DT", abs(t.linear.x - V) < 1e-6, f"v={t.linear.x:.4f}")
check("omega = dyaw/DT", abs(t.angular.z - 0.02 / DT) < 1e-6, f"omega={t.angular.z:.4f}")

print("=== 3. K=10 -> 10ステップ先の行動を使う ===")
acts = {i: np.array([DX, 0.0], np.float32) for i in range(20)}
acts[10] = np.array([DX, 0.05], np.float32)         # 10step先だけ大きく曲がる
t = run(make_node(acts, k=10))
check("omega は10step先の dyaw", abs(t.angular.z - 0.05 / DT) < 1e-6, f"omega={t.angular.z:.4f}")
check("v は「今」の行動から", abs(t.linear.x - V) < 1e-6, f"v={t.linear.x:.4f}")

print("=== 4. K が届いていない -> 現在stepにフォールバック ===")
acts = {0: np.array([DX, 0.03], np.float32)}        # 1件だけ
t = run(make_node(acts, k=10))
check("現在stepの dyaw を使う", abs(t.angular.z - 0.03 / DT) < 1e-6, f"omega={t.angular.z:.4f}")

print("=== 5. Pure Pursuit が不感帯を飛び越える（本題）===")
# 実機で観測された形: 1.4秒(7step)まっすぐ -> その後ゆるやかに左へ戻る
plan = [np.array([DX, 0.0], np.float32)] * 7 + [np.array([DX, 0.012], np.float32)] * 43
acts = {i: a for i, a in enumerate(plan)}
t_raw = run(make_node(acts))                                    # 従来方式
t_short = run(make_node(acts, pp=True, L=1.0))                  # 注視距離が不感帯の中
t_long = run(make_node(acts, pp=True, L=2.5))                   # 注視距離が不感帯の先
print(f"     従来(生dyaw)      omega={t_raw.angular.z:.4f}")
print(f"     PP L=1.0m(不感帯内) omega={t_short.angular.z:.4f}")
print(f"     PP L=2.5m(不感帯外) omega={t_long.angular.z:.4f}")
check("従来方式は omega≈0", abs(t_raw.angular.z) < 1e-9)
check("L=1.0m でも omega≈0（注視距離が短すぎると効かない）", abs(t_short.angular.z) < 1e-6)
check("L=2.5m で omega>0（不感帯を飛び越える）", t_long.angular.z > 0.005)
check("v は PP でも変わらない", abs(t_long.linear.x - V) < 1e-6, f"v={t_long.linear.x:.4f}")

print("=== 6. PP ON で注視点が publish される ===")
n = make_node(acts, pp=True, L=2.5)
run(n)
check("lookahead が publish される", n.lookahead_pub.publish.called)
pt = n.lookahead_pub.publish.call_args[0][0]
check("注視点は 2.5m 以上先", np.hypot(pt.point.x, pt.point.y) >= 2.5,
      f"({pt.point.x:.2f}, {pt.point.y:.2f})")

print("=== 7. PP ON だが経路が足りない -> 生の dyaw にフォールバック ===")
t = run(make_node({0: np.array([DX, 0.04], np.float32)}, pp=True))
check("生の dyaw を使う", abs(t.angular.z - 0.04 / DT) < 1e-6, f"omega={t.angular.z:.4f}")

print("=== 8. 消化されて辞書が伸びない ===")
n = make_node({i: np.array([DX, 0.0], np.float32) for i in range(20)}, k=10)
before = len(n._actions)
n.control_timer_callback()
check("現在stepが辞書から消える", len(n._actions) == before - 1, f"{before} -> {len(n._actions)}")
check("step が進む", n._step == 1)

print("=== 9. クリップ ===")
n = make_node({i: np.array([5.0, 5.0], np.float32) for i in range(20)})
t = run(n)
check("v が上限でクリップ", t.linear.x == 1.0, f"v={t.linear.x}")
check("omega が上限でクリップ", t.angular.z == 1.0, f"omega={t.angular.z}")

print("=== 10. 非自律なら何も publish しない ===")
n = make_node({i: np.array([DX, 0.0], np.float32) for i in range(20)})
n.autonomous_flag = False
n.control_timer_callback()
check("publish されない", not n.cmd_vel_pub.publish.called)

print()
print("FAILED:", fails if fails else "なし")
sys.exit(1 if fails else 0)
