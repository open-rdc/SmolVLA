"""LeRobot データセット(_tc)を NavVLA 形式(episode ごとの jpg / traj_data.pkl / traj_prompt.txt)に書き出す。

OmniVLA-edge を SmolVLA と同じ走行・同じフレーム・同じ言語指示で学習するためのもの。

- 画像: camera1(現在フレーム)の動画を復号して {k}.jpg に書く。SmolVLA が学習で見た画像と同じ
- 言語指示: 各フレームの task をそのまま 1 行ずつ書く
- 位置と向き: 元データの traj_data.pkl があればそれを使う(先頭 length 個)。
  無い場合(復帰データ)は action [dx_body, dyaw] を積分し直す。横方向の移動量は
  _tc 変換で捨てられているので、元データに対して 4.8 秒先で平均 0.11 m ずれる

使い方:
  python export_lerobot_to_navvla.py --root <LeRobot root> --out <出力先> \
      --map splits/holdout_v1_episode_map.json --key ho1 --raw-pkl <元 pkl の親ディレクトリ>
"""
from __future__ import annotations

import argparse
import glob
import json
import pickle
from pathlib import Path

import av
import numpy as np
import pandas as pd
from PIL import Image

CAM = "observation.images.camera1"


def integrate(actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[dx_body, dyaw] の列から各フレームの位置(N,2)と向き(N,)を復元する。"""
    n = len(actions)
    pos = np.zeros((n, 2), dtype=np.float32)
    yaw = np.zeros(n, dtype=np.float32)
    for t in range(n - 1):
        dx, dyaw = actions[t]
        pos[t + 1] = pos[t] + dx * np.array([np.cos(yaw[t]), np.sin(yaw[t])], dtype=np.float32)
        yaw[t + 1] = yaw[t] + dyaw
    return pos, yaw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--map", type=Path, required=True)
    ap.add_argument("--key", required=True, help="map の中のキー (ho1 / heldout1)")
    ap.add_argument("--raw-pkl", type=Path, required=True)
    args = ap.parse_args()

    info = json.load(open(args.root / "meta/info.json"))
    fps = info["fps"]
    emap = json.load(open(args.map))[args.key]
    assert len(emap) == info["total_episodes"], (len(emap), info["total_episodes"])

    eps = pd.concat(
        [pd.read_parquet(p) for p in sorted(glob.glob(str(args.root / "meta/episodes/*/*.parquet")))]
    ).sort_values("episode_index").reset_index(drop=True)
    tasks = pd.read_parquet(args.root / "meta/tasks.parquet")
    task_of = {int(i): t for t, i in zip(tasks.index, tasks["task_index"])}
    data = pd.concat(
        [pd.read_parquet(p, columns=["episode_index", "frame_index", "action", "task_index"])
         for p in sorted(glob.glob(str(args.root / "data/*/*.parquet")))]
    ).sort_values(["episode_index", "frame_index"])
    by_ep = {int(k): g for k, g in data.groupby("episode_index")}

    args.out.mkdir(parents=True, exist_ok=True)
    n_raw = n_int = 0

    # 動画ファイルごとにまとめて先頭から復号する(エピソードごとにシークするより速い)
    eps["vfile"] = list(zip(eps[f"videos/{CAM}/chunk_index"], eps[f"videos/{CAM}/file_index"]))
    for (chunk, fidx), group in eps.groupby("vfile", sort=True):
        path = args.root / info["video_path"].format(video_key=CAM, chunk_index=chunk, file_index=fidx)
        spans = [(float(r[f"videos/{CAM}/from_timestamp"]), int(r["length"]), int(r["episode_index"]))
                 for _, r in group.iterrows()]
        written: dict[int, set[int]] = {e: set() for _, _, e in spans}
        for _, _, e in spans:
            (args.out / f"ep{e:05d}").mkdir(exist_ok=True)
        with av.open(str(path)) as container:
            for frame in container.decode(video=0):
                t = float(frame.pts * frame.time_base)
                for start, length, e in spans:
                    k = round((t - start) * fps)
                    if 0 <= k < length:
                        # 復号したそばから書く(1ファイル分を溜めると数GBになる)
                        frame.to_image().save(args.out / f"ep{e:05d}" / f"{k}.jpg", quality=95)
                        written[e].add(k)
                        break

        for start, length, e in spans:
            entry = emap[e]
            assert entry["length"] == length, (e, entry["length"], length)
            assert written[e] == set(range(length)), f"episode {e}: decoded {len(written[e])} / {length}"
            g = by_ep[e]
            assert len(g) == length
            ep_dir = args.out / f"ep{e:05d}"
            prompts = [task_of[int(i)] for i in g["task_index"]]
            (ep_dir / "traj_prompt.txt").write_text("\n".join(prompts), encoding="utf-8")
            if entry["raw"] is not None:
                d = pickle.load(open(args.raw_pkl / entry["raw"] / "traj_data.pkl", "rb"))
                pos = np.asarray(d["position"], dtype=np.float32)[:length]
                yaw = np.asarray(d["yaw"], dtype=np.float32)[:length]
                n_raw += 1
            else:
                pos, yaw = integrate(np.stack(g["action"].to_numpy()).astype(np.float32))
                n_int += 1
            with open(ep_dir / "traj_data.pkl", "wb") as f:
                pickle.dump({"position": pos, "yaw": yaw, "source": entry["raw"] or "integrated"}, f)
        print(f"[export] {path.name}: {len(spans)} episodes", flush=True)

    print(f"EXPORT_DONE episodes={n_raw + n_int} raw_pose={n_raw} integrated_pose={n_int}")


if __name__ == "__main__":
    main()
