#!/usr/bin/env python3
"""create_data.py を ROS スタブ上で実際に動かし、出力が変換器の期待する形か検証する。

`/flag` トグル -> フレーム収集 -> `/flag` トグル の一連を実物のコールバックで再現し、
できた episode ディレクトリを training/data/lerobot_dataset.py の load_episode で
読み直せることまで確認する。
"""
import shutil
import sys
import tempfile
import types
from pathlib import Path
from unittest import mock

import numpy as np

REPO = Path(__file__).resolve().parents[2]      # SmolVLA/


# ---- ROS スタブ ------------------------------------------------------
class StubNode:
    def __init__(self, *_a, **_k):
        self._params = {}

    def declare_parameter(self, name, default):
        self._params[name] = default

    def get_parameter(self, name):
        return types.SimpleNamespace(value=self._params[name])

    def create_subscription(self, *_a, **_k):
        return None

    def create_timer(self, *_a, **_k):
        return None

    def get_logger(self):
        return mock.Mock()


for name in ["rclpy", "rclpy.node", "rclpy.qos", "nav_msgs", "nav_msgs.msg",
             "sensor_msgs", "sensor_msgs.msg", "std_msgs", "std_msgs.msg"]:
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["rclpy.node"].Node = StubNode
sys.modules["rclpy.qos"].qos_profile_system_default = None
sys.modules["nav_msgs.msg"].Odometry = object
sys.modules["sensor_msgs.msg"].Image = object
sys.modules["std_msgs.msg"].Empty = object
sys.modules["std_msgs.msg"].String = object

sys.path.insert(0, str(REPO / "deployment"))
from smolvla_nav import create_data as cd  # noqa: E402

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")
    if not cond:
        fails.append(name)


def fake_image(h=480, w=640, seed=0):
    """bgr8 の sensor_msgs/Image 相当（image_msg_to_bgr が読める形）。"""
    rng = np.random.default_rng(seed)
    data = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    return types.SimpleNamespace(encoding="bgr8", height=h, width=w, step=w * 3,
                                 data=data.tobytes())


def fake_odom(x, y, yaw):
    q = types.SimpleNamespace(x=0.0, y=0.0, z=np.sin(yaw / 2), w=np.cos(yaw / 2))
    pose = types.SimpleNamespace(position=types.SimpleNamespace(x=x, y=y), orientation=q)
    return types.SimpleNamespace(pose=types.SimpleNamespace(pose=pose))


tmp = Path(tempfile.mkdtemp(prefix="create_data_test_"))
try:
    node = cd.DataCreator.__new__(cd.DataCreator)
    StubNode.__init__(node)
    node._params = {}
    # __init__ を通す（保存先だけ差し替える）
    orig_declare = node.declare_parameter

    def declare(name, default):
        orig_declare(name, str(tmp) if name == "output_dir" else default)

    node.declare_parameter = declare
    cd.DataCreator.__init__(node)

    print("=== 1. 開始前は何も書かない ===")
    node.timer_callback()
    check("エピソードディレクトリ未作成", node.current_episode_dir is None)

    print("=== 2. /flag で収録開始 -> 10フレーム ===")
    node.flag_callback(None)
    check("収録中", node.collect_flag)
    check("episode01 が作られる", node.current_episode_dir.name == "episode01",
          node.current_episode_dir.name)

    node.timer_callback()      # odom/image がまだ無い -> 記録しない
    check("odom/image 未受信なら記録しない", node.current_sample_index == 0)

    N = 10
    for i in range(N):
        node.image_callback(fake_image(seed=i))
        # ±pi をまたぐ yaw にして unwrap が効いているか見る
        node.odom_callback(fake_odom(x=0.2 * i, y=0.0, yaw=3.10 + 0.05 * i))
        node.prompt_callback(types.SimpleNamespace(data=f"turn left at the corner {i}"))
        node.timer_callback()
    check(f"{N} フレーム記録", node.current_sample_index == N, f"{node.current_sample_index}")

    print("=== 3. /flag で停止 -> 保存 ===")
    node.flag_callback(None)
    check("収録停止", not node.collect_flag)
    ep = tmp / node.dataset_dir.name / "episode01"
    jpgs = sorted(ep.glob("*.jpg"), key=lambda p: int(p.stem))
    check(f"jpg が {N} 枚", len(jpgs) == N, f"{len(jpgs)}")
    check("連番 0..N-1", [int(p.stem) for p in jpgs] == list(range(N)))
    check("traj_data.pkl がある", (ep / "traj_data.pkl").exists())
    check("traj_prompt.txt がある", (ep / "traj_prompt.txt").exists())

    print("=== 4. 画像は 224x224 BGR ===")
    import cv2
    im = cv2.imread(str(jpgs[0]))
    check("224x224x3", im.shape == (224, 224, 3), str(im.shape))

    print("=== 5. 変換器の load_episode で読み直せる ===")
    sys.path.insert(0, str(REPO / "training" / "data"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("ldset", REPO / "training/data/lerobot_dataset.py")
    ldset = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ldset)
    pos, yaw, prompts = ldset.load_episode(ep)
    check("position shape", pos.shape == (N, 2), str(pos.shape))
    check("yaw shape", yaw.shape == (N,), str(yaw.shape))
    check("prompt 行数 = フレーム数", len(prompts) == N, f"{len(prompts)}")
    check("prompt の中身が一致", prompts[3] == "turn left at the corner 3", prompts[3])
    check("position が記録どおり", np.allclose(pos[:, 0], 0.2 * np.arange(N)))

    print("=== 6. yaw が unwrap されている（±piをまたいでも連続）===")
    check("yaw に 2pi の飛びが無い", np.all(np.abs(np.diff(yaw)) < np.pi),
          f"max|diff|={np.abs(np.diff(yaw)).max():.3f}")
    check("yaw は単調増加", np.all(np.diff(yaw) > 0))

    print("=== 7. エピソード検出（変換器と同じ条件）===")
    ds = tmp / node.dataset_dir.name
    found = sorted(p for p in ds.iterdir() if p.is_dir() and (p / "traj_data.pkl").exists())
    check("1エピソード検出", len(found) == 1, str([p.name for p in found]))

    print("=== 8. 2本目 + 空エピソードは削除される ===")
    node.flag_callback(None)                      # episode02 開始
    node.image_callback(fake_image(seed=99))
    node.odom_callback(fake_odom(1.0, 1.0, 0.0))
    node.timer_callback()
    node.flag_callback(None)                      # 保存
    check("episode02 が保存される", (ds / "episode02").exists())
    node.flag_callback(None)                      # episode03 開始（フレーム0のまま）
    node.flag_callback(None)                      # 停止
    check("空の episode03 は削除される", not (ds / "episode03").exists())

    print("=== 9. Ctrl-C で収録中のエピソードが確定する ===")
    node.flag_callback(None)                      # episode04 開始
    node.image_callback(fake_image(seed=7))
    node.odom_callback(fake_odom(2.0, 2.0, 0.5))
    node.timer_callback()
    node.save_data()                              # Ctrl-C 相当
    check("episode04 の traj_data.pkl が書かれる", (ds / "episode04" / "traj_data.pkl").exists())

    print()
    print("FAILED:", fails if fails else "なし")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

sys.exit(1 if fails else 0)
