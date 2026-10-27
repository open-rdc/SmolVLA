#!/usr/bin/env python3
"""rosbag(1周ぶんの手動走行)を、注釈用の長い 1 エピソードに変換する。

create_data.py と同じく 5 Hz で「その時点の最新の画像とオドメトリ」を 1 組ずつ取り出す。
create_data.py との違い:
  - 画像はクロップも縮小もしない。bag の JPEG(640x480 魚眼)をそのまま書き出す
    (魚眼の画角を残すため。縮小は LeRobot 変換時に行う)
  - /flag が bag に無いので、1 本の bag を 1 エピソードにする。指示ごとの区切りは
    注釈後に split_lap_by_prompt.py で行う
  - オドメトリの飛び(1 ステップで MAX_STEP_M 超)や途切れ(MAX_GAP_S 超)がある bag は、
    それらで区切った中で最も長い区間だけを使う(LIO の起動直後のリセット対策)
  - 停止(STOP_MIN_FRAMES フレーム以上続くもの)は削除する(番号は詰める)。--keep-stops なら停止は残す
  - オドメトリの更新が止まったフレーム(位置が前と完全に同じ)と、1 フレームで JUMP_M を
    超えて跳ぶフレームも削除する(画像と位置の対応がずれているため)

変換済みの周回から停止だけを削除するときは --drop-stops に episode01 を並べる。
traj_prompt*.txt と review_state.json があれば、同じフレームを削る:
    python training/data/bag_to_lap_episode.py --drop-stops ~/tsudanuma_data/laps/*/episode01

出力:
    <out_root>/<bag名>/episode01/0.jpg ... N-1.jpg
                                /traj_data.pkl   {"position": (N,2) float32, "yaw": (N,) float32 unwrap済}
                                /timestamps.npy  (N,) int64 [ns]  bag 時刻

    python training/data/bag_to_lap_episode.py ~/tsudanuma_bags/tsudanuma_20261010_142912 \
        --out-root ~/tsudanuma_data/laps
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path

import cv2
import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

FPS = 5
INTERVAL_NS = int(1e9 / FPS)
MAX_STEP_M = 1.0   # オドメトリ 1 メッセージ(約 0.1 s)でこれ以上動いたら飛びとみなす
MAX_GAP_S = 1.0    # オドメトリがこれ以上途切れたら区切る
IMAGE_TOPIC = "/image_raw/compressed"
RAW_IMAGE_TOPIC = "/image_raw"  # 圧縮画像が無い bag 用(rgb8)。書き出し時に JPEG にする
ODOM_TOPIC = "/Odometry"
STOP_V = 0.05          # [m/s] これ未満
STOP_W = 0.05          # [rad/s] かつこれ未満のフレームを停止とみなす
STOP_MIN_FRAMES = 1    # 停止がこれ以上続いたら削除する(1 = 停止はすべて削除)
JUMP_M = 0.5           # 1 フレーム(0.2 s)でこれを超えて動いたら跳びとみなす(2.5 m/s)
FROZEN_M = 1e-4        # 前のフレームからの移動がこれ未満なら更新停止とみなす


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def valid_window(reader: AnyReader) -> tuple[int, int, list[str]]:
    """オドメトリの飛び・途切れで区切った区間のうち、最長の区間の [開始, 終了] 時刻を返す。"""
    conns = [c for c in reader.connections if c.topic == ODOM_TOPIC]
    ts, xy = [], []
    for c, t, raw in reader.messages(connections=conns):
        p = reader.deserialize(raw, c.msgtype).pose.pose.position
        ts.append(t)
        xy.append((p.x, p.y))
    ts, xy = np.array(ts), np.array(xy)
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    gap = np.diff(ts) / 1e9
    cut = np.where((step > MAX_STEP_M) | (gap > MAX_GAP_S))[0]  # cut[i] と cut[i]+1 の間で区切る
    bounds = [0, *(cut + 1).tolist(), len(ts)]
    seg = max(zip(bounds[:-1], bounds[1:]), key=lambda b: b[1] - b[0])
    notes = [f"{(ts[i + 1] - ts[0]) / 1e9:.1f}s: step={step[i]:.2f}m gap={gap[i]:.2f}s" for i in cut]
    return int(ts[seg[0]]), int(ts[seg[1] - 1]), notes


def stop_mask(position: np.ndarray, yaw: np.ndarray, keep_stops: bool = False) -> np.ndarray:
    """STOP_MIN_FRAMES 以上続く停止と、オドメトリの更新停止・跳びのフレームを True にする。
    keep_stops なら停止は残し、更新停止・跳びだけを True にする(つくばの横断歩道前の停止を残すため。
    停止中もオドメトリは揺れるので、位置が完全に同じになる更新停止とは区別できる)。"""
    step = np.r_[np.nan, np.linalg.norm(np.diff(position, axis=0), axis=1)]
    v = np.r_[np.inf, step[1:] * FPS]
    w = np.r_[np.inf, np.abs(np.diff(yaw)) * FPS]
    still = (v < STOP_V) & (w < STOP_W)
    still[0] = still[1] if len(still) > 1 else False  # 先頭は速度が無いので次のフレームに合わせる
    mask = np.zeros(len(still), dtype=bool)
    i = len(still) if keep_stops else 0
    while i < len(still):
        j = i
        while j < len(still) and still[j] == still[i]:
            j += 1
        if still[i] and j - i >= STOP_MIN_FRAMES:
            mask[i:j] = True
        i = j
    mask |= (step < FROZEN_M) | (step > JUMP_M)  # nan(先頭)は False
    return mask


def drop_stops(ep_dir: Path, keep_stops: bool = False) -> None:
    """ep_dir から長い停止のフレームを削除し、画像番号を 0 から詰め直す。"""
    with (ep_dir / "traj_data.pkl").open("rb") as f:
        traj = pickle.load(f)
    stamps = np.load(ep_dir / "timestamps.npy")
    n = len(stamps)
    drop = stop_mask(traj["position"], traj["yaw"], keep_stops)
    keep = np.where(~drop)[0]
    if len(keep) == n:
        print(f"{ep_dir}: 削除するフレームなし")
        return
    # 行ごとのファイルを先に検査してから書き換える(途中で失敗して不整合にならないように)
    line_files = sorted(ep_dir.glob("traj_prompt*.txt"))
    lines = {p: p.read_text(encoding="utf-8").split("\n") for p in line_files}
    state = ep_dir / "review_state.json"
    reviewed = json.loads(state.read_text())["reviewed"] if state.exists() else None
    for p, ls in lines.items():
        assert len(ls) == n, f"{p}: {len(ls)} 行 != {n} フレーム"
    assert reviewed is None or len(reviewed) == n
    for i in np.where(drop)[0]:
        (ep_dir / f"{i}.jpg").unlink()
    for j, i in enumerate(keep):
        if i != j:
            (ep_dir / f"{i}.jpg").rename(ep_dir / f"{j}.jpg")
    with (ep_dir / "traj_data.pkl").open("wb") as f:
        pickle.dump({"position": traj["position"][keep], "yaw": traj["yaw"][keep]}, f)
    np.save(ep_dir / "timestamps.npy", stamps[keep])
    for p, ls in lines.items():
        p.write_text("\n".join(ls[i] for i in keep), encoding="utf-8")
    if reviewed is not None:
        state.write_text(json.dumps({"reviewed": [reviewed[i] for i in keep]}))
    print(f"{ep_dir}: 停止・オドメトリ異常 {n - len(keep)} フレームを削除 ({n} -> {len(keep)})")


def convert(bag: Path, out_root: Path, keep_stops: bool = False) -> None:
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    ep_dir = out_root / bag.name / "episode01"
    ep_dir.mkdir(parents=True, exist_ok=False)
    with AnyReader([bag], default_typestore=typestore) as reader:
        t_start, t_end, notes = valid_window(reader)
        conns = [c for c in reader.connections if c.topic in (IMAGE_TOPIC, RAW_IMAGE_TOPIC, ODOM_TOPIC)]
        if any(c.topic == IMAGE_TOPIC for c in conns):
            conns = [c for c in conns if c.topic != RAW_IMAGE_TOPIC]
        latest_img = None  # JPEG の bytes、または未圧縮の Image メッセージ
        latest_pose: tuple[float, float, float] | None = None
        next_tick = t_start
        positions, yaws, stamps = [], [], []
        for c, t, raw in reader.messages(connections=conns, start=t_start, stop=t_end + 1):
            # 時刻 t のメッセージを取り込む前に、t までに到来したティックを確定させる
            while t >= next_tick:
                if latest_img is not None and latest_pose is not None:
                    i = len(positions)
                    if not isinstance(latest_img, bytes):
                        m = latest_img
                        rgb = np.frombuffer(m.data, np.uint8).reshape(m.height, m.step)[:, : m.width * 3]
                        latest_img = cv2.imencode(".jpg", cv2.cvtColor(rgb.reshape(m.height, m.width, 3), cv2.COLOR_RGB2BGR))[1].tobytes()
                    (ep_dir / f"{i}.jpg").write_bytes(latest_img)
                    positions.append(latest_pose[:2])
                    yaws.append(latest_pose[2])
                    stamps.append(next_tick)
                next_tick += INTERVAL_NS
            msg = reader.deserialize(raw, c.msgtype)
            if c.topic == IMAGE_TOPIC:
                latest_img = bytes(msg.data)
            elif c.topic == RAW_IMAGE_TOPIC:
                assert msg.encoding == "rgb8", msg.encoding
                latest_img = msg
            else:
                p, q = msg.pose.pose.position, msg.pose.pose.orientation
                latest_pose = (p.x, p.y, yaw_from_quaternion(q.x, q.y, q.z, q.w))

    positions = np.asarray(positions, dtype=np.float32)
    yaws = np.unwrap(np.asarray(yaws, dtype=np.float64)).astype(np.float32)
    with (ep_dir / "traj_data.pkl").open("wb") as f:
        pickle.dump({"position": positions, "yaw": yaws}, f)
    np.save(ep_dir / "timestamps.npy", np.asarray(stamps, dtype=np.int64))
    length = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
    print(f"{bag.name}: {len(positions)} フレーム ({len(positions) / FPS / 60:.1f} 分, {length:.0f} m) -> {ep_dir}")
    for n in notes:
        print(f"    オドメトリの区切り {n}")
    drop_stops(ep_dir, keep_stops)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bags", nargs="*", type=Path)
    ap.add_argument("--out-root", type=Path)
    ap.add_argument("--drop-stops", nargs="+", type=Path, default=[],
                    help="変換済みの episode01 から停止だけを削除する")
    ap.add_argument("--keep-stops", action="store_true",
                    help="停止は残し、オドメトリの更新停止・跳びだけを削除する")
    args = ap.parse_args()
    for ep_dir in args.drop_stops:
        drop_stops(ep_dir.expanduser(), args.keep_stops)
    if args.bags and args.out_root is None:
        ap.error("bag を変換するときは --out-root が必要")
    for bag in args.bags:
        convert(bag.expanduser(), args.out_root.expanduser(), args.keep_stops)


if __name__ == "__main__":
    main()
