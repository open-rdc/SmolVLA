#!/usr/bin/env python3
"""pure_pursuit.py の単体テスト（ROS不要）。"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "smolvla_nav"))
from pure_pursuit import find_lookahead_point, integrate_path, pure_pursuit_omega

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")
    if not cond:
        fails.append(name)


DX = 0.195  # 実データの1ステップ移動量 [m] (=約1.0 m/s @ dt=0.2)

print("=== 1. 直進 ===")
straight = np.array([[DX, 0.0]] * 20)
p = integrate_path(straight)
check("x が等間隔に増える", np.allclose(np.diff(p[:, 0]), DX), f"x[-1]={p[-1,0]:.3f}")
check("y は 0 のまま", np.allclose(p[:, 1], 0.0))
check("theta は 0 のまま", np.allclose(p[:, 2], 0.0))
g = find_lookahead_point(p, 2.5)
check("注視点は 2.5m 以上先", g[0] >= 2.5, f"goal=({g[0]:.3f}, {g[1]:.3f})")
check("直進なら omega=0", abs(pure_pursuit_omega(1.0, *g)) < 1e-9)

print("=== 2. 学習データの action 構成の厳密な逆変換になっているか ===")
# lerobot_dataset.py: dxy_body = to_body_frame(pos[t+1]-pos[t], yaw[t]) の x 成分 + dyaw。
# to_body_frame は R(-yaw) 相当なので、非ホロノミック(dy_body=0)な軌跡を作って往復させる。
rng = np.random.default_rng(0)
n = 60
step_len = 0.15 + 0.1 * rng.random(n)               # 各ステップの前進量
dyaws = 0.15 * (rng.random(n) - 0.5)               # 各ステップの旋回量

yaw = np.concatenate([[0.0], np.cumsum(dyaws)])     # yaw[0]=0 で base_link と揃える
pos = np.zeros((n + 1, 2))
for t in range(n):
    # 現在の向きに step_len だけ進む = dy_body が厳密に 0 になる軌跡
    pos[t + 1] = pos[t] + step_len[t] * np.array([math.cos(yaw[t]), math.sin(yaw[t])])


def to_body_frame(delta_xy_global, yaw_t):          # lerobot_dataset.py と同一実装
    c, s = math.cos(yaw_t), math.sin(yaw_t)
    return delta_xy_global.dot(np.array([[c, -s], [s, c]]))


actions = np.array([[to_body_frame(pos[t + 1] - pos[t], yaw[t])[0], dyaws[t]] for t in range(n)])
p = integrate_path(actions)
check("位置を厳密に復元する", np.allclose(p[:, :2], pos[1:], atol=1e-9),
      f"max誤差={np.abs(p[:, :2] - pos[1:]).max():.2e}")
check("yaw を厳密に復元する", np.allclose(p[:, 2], yaw[1:], atol=1e-9),
      f"max誤差={np.abs(p[:, 2] - yaw[1:]).max():.2e}")

print("=== 2b. 真円上の注視点なら omega = v/R（注視距離によらず） ===")
R, v = 9.75, 1.0
# 原点で x 軸に接し中心 (0,R) の円: x=R sinφ, y=R(1-cosφ)
phi = np.linspace(0.0, 1.2, 400)
circle = np.stack([R * np.sin(phi), R * (1 - np.cos(phi)), phi], axis=1)
for L in (1.0, 2.0, 2.5, 3.0):
    g = find_lookahead_point(circle, L)
    om = pure_pursuit_omega(v, *g)
    check(f"omega=v/R (L={L})", math.isclose(om, v / R, rel_tol=1e-9),
          f"omega={om:.6f} v/R={v / R:.6f}")

print("=== 3. 符号 ===")
check("注視点が左(y>0) -> omega>0", pure_pursuit_omega(1.0, 2.0, 0.5) > 0)
check("注視点が右(y<0) -> omega<0", pure_pursuit_omega(1.0, 2.0, -0.5) < 0)
check("v=0 なら omega=0", pure_pursuit_omega(0.0, 2.0, 0.5) == 0.0)

print("=== 4. 端のケース ===")
check("空 -> None", find_lookahead_point(np.empty((0, 3)), 2.5) is None)
short = integrate_path(np.array([[DX, 0.0]] * 3))     # 0.585m しかない
g = find_lookahead_point(short, 2.5)
check("経路が短い -> 終端を返す", np.allclose(g, short[-1, :2]), f"goal=({g[0]:.3f}, {g[1]:.3f})")
check("注視点が原点なら omega=0", pure_pursuit_omega(1.0, 0.0, 0.0) == 0.0)

print("=== 5. 不感帯を飛び越せるか（本題） ===")
# 「1.5秒(=7step)まっすぐ -> その後ゆるやかに左へ戻る」という実機で観測された計画を再現
dead = [[DX, 0.0]] * 7
turn = [[DX, 0.012]] * 43
plan = integrate_path(np.array(dead + turn))
v = DX / 0.2
om_raw = plan[0, 2] / 0.2            # 生の dyaw[0]/DT = 現行方式
check("現行方式(生dyaw)はほぼ0", abs(om_raw) < 1e-9, f"omega_raw={om_raw:.5f}")
for L in (1.0, 2.5):
    g = find_lookahead_point(plan, L)
    om = pure_pursuit_omega(v, *g)
    print(f"     L={L}m -> goal=({g[0]:.2f},{g[1]:.2f}) omega={om:.4f} rad/s")
g_short = find_lookahead_point(plan, 1.0)
g_long = find_lookahead_point(plan, 2.5)
check("L=1.0m は不感帯の中 -> omega≈0",
      abs(pure_pursuit_omega(v, *g_short)) < 1e-6)
check("L=2.5m は不感帯の先 -> omega>0",
      pure_pursuit_omega(v, *g_long) > 0.01)

print()
print("FAILED:", fails if fails else "なし")
sys.exit(1 if fails else 0)
