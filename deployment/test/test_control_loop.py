#!/usr/bin/env python3
"""navigation.py の control_timer_callback を、ROS/torch をスタブして実際に動かすテスト。

実物のメソッドを呼ぶので、分岐ロジック（+K のフォールバック、停止指令、辞書のGC、
クリップ）がそのまま検証される。Pure Pursuit による操舵の決定は path_follower.py に
分離したので、そちらのテストは test_path_follower.py を参照。
"""
import sys
from pathlib import Path
import threading
import types
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


def make_node(actions, step=0, k=0):
    """control_timer_callback を呼べる最小のスタブ node を作る。"""
    n = object.__new__(nav.SmolVLANavigationNode)
    n.autonomous_flag = True
    n._queue_lock = threading.Lock()
    n._step = step
    n._actions = dict(actions)
    n.linear_max_vel = 1.0
    n.angular_max_vel = 1.0
    n.path_frame_id = "base_link"
    params = {"step_lookahead": k}
    n.get_parameter = lambda name: types.SimpleNamespace(value=params[name])
    n.cmd_vel_pub = mock.Mock()
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

print("=== 2. K=0 -> 従来どおり生の dyaw ===")
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

print("=== 5. 消化されて辞書が伸びない ===")
n = make_node({i: np.array([DX, 0.0], np.float32) for i in range(20)}, k=10)
before = len(n._actions)
n.control_timer_callback()
check("現在stepが辞書から消える", len(n._actions) == before - 1, f"{before} -> {len(n._actions)}")
check("step が進む", n._step == 1)

print("=== 6. クリップ ===")
n = make_node({i: np.array([5.0, 5.0], np.float32) for i in range(20)})
t = run(n)
check("v が上限でクリップ", t.linear.x == 1.0, f"v={t.linear.x}")
check("omega が上限でクリップ", t.angular.z == 1.0, f"omega={t.angular.z}")

print("=== 7. 非自律なら何も publish しない ===")
n = make_node({i: np.array([DX, 0.0], np.float32) for i in range(20)})
n.autonomous_flag = False
n.control_timer_callback()
check("publish されない", not n.cmd_vel_pub.publish.called)

print()
print("FAILED:", fails if fails else "なし")
sys.exit(1 if fails else 0)
