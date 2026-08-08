#!/usr/bin/env python3
"""Pure Pursuit の幾何計算（ROS・torch に依存しない純粋関数）。

navigation.py から使う。ROS に依存しないので単体でテストできる。

座標系は REP-103 の base_link 準拠:
  x = 前方 / y = 左 / theta = 反時計回り正
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


def integrate_path(actions: np.ndarray) -> np.ndarray:
    """body 系の 1 ステップ増分列を、現在姿勢を原点とする姿勢列に積分する。

    navigation.py の _publish_pred_path() と同じ積分を行う（あちらは可視化用、
    こちらは制御用）。

    Args:
        actions: (N, 2) の [[dx_body, dyaw], ...]。学習した action そのもの。

    Returns:
        (N, 3) の [[x, y, theta], ...]。i 番目は「i+1 ステップ実行し終えた時点」の姿勢。
        現在姿勢 (0,0,0) は含まない。
    """
    actions = np.asarray(actions, dtype=np.float64).reshape(-1, 2)
    poses = np.empty((len(actions), 3), dtype=np.float64)

    x = y = theta = 0.0
    for i, (dx_body, dyaw) in enumerate(actions):
        # 各ステップは「現在の向きに dx_body 進んでから dyaw 回る」。
        x += dx_body * math.cos(theta)
        y += dx_body * math.sin(theta)
        theta += dyaw
        poses[i] = (x, y, theta)
    return poses


def find_lookahead_point(poses: np.ndarray, lookahead: float) -> Optional[np.ndarray]:
    """原点からの直線距離が lookahead 以上になる最初の点を返す。

    経路が lookahead より短い場合は最遠点（経路の終端）を返す。これは Pure Pursuit
    の慣例で、注視点が無いより終端を追う方が破綻しないため。

    Args:
        poses: (N, 3) の [[x, y, theta], ...]（integrate_path の出力）。
        lookahead: 前方注視距離 [m]。

    Returns:
        (2,) の [x, y]。poses が空なら None。
    """
    poses = np.asarray(poses, dtype=np.float64).reshape(-1, 3)
    if len(poses) == 0:
        return None

    xy = poses[:, :2]
    dist = np.hypot(xy[:, 0], xy[:, 1])
    idx = np.flatnonzero(dist >= lookahead)
    if len(idx) == 0:
        # 条件を満たす点が無い = 経路が注視距離より短い -> 終端を使う。
        return xy[-1]
    return xy[idx[0]]


def pure_pursuit_omega(v: float, goal_x: float, goal_y: float) -> float:
    """前方注視点を通る円弧を描くための角速度 [rad/s] を返す。

    ロボット座標系で注視点 (x, y) を通る円弧の曲率は kappa = 2y / (x^2 + y^2)。
    角速度は omega = v * kappa。y > 0（注視点が左）なら omega > 0（左旋回）。

    「計画の初速(dyaw[0])」ではなく「計画の行き先」から操舵量を決めるので、
    計画の先頭にある不感帯（まっすぐな前置き）を飛び越せる。これが復帰動作を
    強くする狙い。注視距離が不感帯より短いと不感帯の中を見ることになり、
    生の dyaw を使うのと変わらなくなる点に注意。

    Args:
        v:      並進速度 [m/s]。
        goal_x: 注視点の前方距離 [m]。
        goal_y: 注視点の左方向オフセット [m]。

    Returns:
        角速度 [rad/s]。注視点が原点に一致する場合は 0.0。
    """
    d_sq = goal_x * goal_x + goal_y * goal_y
    if d_sq < 1e-9:
        return 0.0
    curvature = 2.0 * goal_y / d_sq
    return v * curvature
