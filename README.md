# SmolVLA

## Overview

SmolVLA は、HuggingFace [`lerobot`](https://github.com/huggingface/lerobot) の
[SmolVLA](https://huggingface.co/lerobot/smolvla_base) を用いた、差動2輪移動ロボット向けナビゲーション実装です。
本リポジトリには、以下の 2 つの機能が含まれます。

- `SmolVLA` のファインチューニング(移動ロボットのナビゲーション向けに action/state を再定義)
- ROS 2 ノードによるナビゲーション推論

`lerobot` 本体はサブモジュールとして管理され、本リポジトリ側ではデータセット変換、学習ジョブ、
可視化スクリプト、ROS 2 ノードを提供します。

## Installation

`lerobot` は Python≥3.12 必須です。ROS 2 Humble は Python 3.10 のため、**学習用venvとROS 2環境を分けて**セットアップします。

### 学習環境

```bash
git clone --recurse-submodules git@github.com:open-rdc/SmolVLA.git
cd SmolVLA
uv venv --python 3.12 ~/.venvs/smolvla
uv pip install -e "./lerobot[smolvla,dataset,training]" --python ~/.venvs/smolvla/bin/python
```

`uv venv` で作った venv には `pip` が入っていないため、パッケージ追加は必ず `uv pip` を使ってください
(素の `pip` を使うとシステム側の Python にフォールバックします)。

### ROS 2 環境(推論)

ROS 2 Humble のワークスペースに本リポジトリの `deployment/` を配置してビルドします。

```bash
cd ~/ros2_ws/src
ln -s <このリポジトリ>/deployment smolvla_nav   # または deployment/ ごとサブモジュール化
cd ~/ros2_ws
colcon build --packages-select smolvla_nav
source install/setup.bash
```

推論には学習済みチェックポイント(`config.json` / `model.safetensors` / pre・post processor 一式)が必要です。
配置先は [`deployment/smolvla_nav/navigation.py`](https://github.com/open-rdc/SmolVLA/blob/main/deployment/smolvla_nav/navigation.py) の `DEFAULT_CKPT` を参照してください。

## Training

### 1. 事前準備

- `lerobot` サブモジュールを取得する(`git submodule update --init`)
- 学習用データセットを [Dataset](#dataset) の形式で用意し、`LeRobotDataset` 化する
- TensorBoard で loss を見る場合は学習venvに `tensorboard` を追加インストールする

### 2. 実行コマンド

```bash
~/.venvs/smolvla/bin/lerobot-train \
  --policy.path=lerobot/smolvla_base --policy.push_to_hub=false \
  --dataset.repo_id=open-rdc/tsudanuma_nav6 --dataset.root=<abs path> \
  --rename_map='{"observation.images.front":"observation.images.camera1"}' \
  --wandb.enable=false
```

**確定した学習レシピ**: VLM(SmolVLM2)は事前学習を維持したまま凍結し、action expertのみランダム初期化して学習する
(`--policy.type=smolvla --policy.load_vlm_weights=true`、`--policy.path` は指定しない)。
理由は [Findings](#findings) を参照してください。

既存チェックポイントからの継続ファインチューンを行う場合は `--policy.path=<ckptのpretrained_modelディレクトリ>` を指定し、
元のcosineスケジュールへ完全restart(peak lrへ戻す)するのではなく、**peakを元の1/5程度に抑えた短いwarmup+cosine decay**
で継続します(収束済みモデルを壊さないため)。

### 3. TensorBoard

`lerobot` は TensorBoard 出力を持たず、`.out` ログ(wandb無効時のstdoutログ)のみを出力します。
[`training/loss_to_tb.py`](https://github.com/open-rdc/SmolVLA/blob/main/training/loss_to_tb.py) で後処理してTensorBoard event化します。

```bash
~/.venvs/smolvla/bin/python training/loss_to_tb.py --segment <jobname>.out:0 --logdir tb_all
~/.venvs/smolvla/bin/tensorboard --logdir tb_all --host 0.0.0.0 --port 6007
```

記録される主な scalar:

- `train/loss`
- `train/grad_norm`
- `train/lr`
- `train/epoch`
- `val/loss`(`--dataset.eval_split` 有効時のみ)

resumeで学習が複数の `.out` に分かれた場合は `--segment FILE:OFFSET` を時系列順に複数指定すると1本の連続曲線になり、
`--follow` で最後のsegmentの追記をライブ監視できます。既存ckptからの継続学習フェーズ(新規run)はepochカウンタが
0から数え直されますが、直前segmentの最終epochより小さい値で始まっていれば自動でoffsetを足すため、境界をまたいでも
連続した曲線になります。

## Dataset

学習には `LeRobotDataset` 形式のデータセットを使用します。要求されるデータ項目は以下の通りです。

| 項目 | 内容 |
|---|---|
| `observation.images.front` | 224×224 RGB(学習時は `camera1` にrename) |
| `observation.state` | `[v, ω]`(前フレームの増分÷dt、学習時ノイズ付加でcopycat対策) |
| `action` | `[Δx_body, Δyaw]`(差動2輪のため `Δy_body` は非ホロノミックで冗長、使わない) |
| `task` | per-frameの言語指示文字列 |
| fps | 5 (`dt = 0.2s`) |

SmolVLA既定は3カメラ・state/action 6次元を期待しますが、1カメラ・2次元のまま32次元パディングで吸収して使っています。

## Navigation

### 概要

ROS 2 ノード `navigation_node` は、画像と言語指示を購読し、`SmolVLAPolicy` で推定した行動チャンクの先頭数stepを
`geometry_msgs/Twist` として出力します(receding horizon、次tickで撮り直し)。`place_prompt_node` は走行データから
作ったトポロジカルマップ上で自己位置推定を行い、現在位置に対応する言語指示を `/prompt` に自動配信します。

ROS 2(Humble, Python 3.10)と `lerobot`(Python 3.12)はプロセスを分けず、同一プロセス内で直接importする構成です。

起動ファイル:

- [`deployment/launch/smolvla_nav.launch.py`](https://github.com/open-rdc/SmolVLA/blob/main/deployment/launch/smolvla_nav.launch.py)

### 起動方法

```bash
ros2 launch smolvla_nav smolvla_nav.launch.py                    # トポロジカルマップで自己位置推定→自動プロンプト切替
ros2 launch smolvla_nav smolvla_nav.launch.py use_toponav:=false # 固定プロンプトのみ(place_prompt_nodeを止める)
```

トポロジカルマップの作成:

```bash
ros2 run smolvla_nav create_topomap --ros-args -p data_dir:=<走行データ> -p output:=deployment/config/topomap
```

### Topic 一覧

| Topic | 型 | 方向 | Node | 内容 |
|---|---|---|---|---|
| `/image_raw` | `sensor_msgs/msg/Image` | Subscribe | navigation_node, place_prompt_node | 現在観測画像 |
| `/autonomous` | `std_msgs/msg/Bool` | Subscribe | navigation_node | 自律動作の有効/無効 |
| `/prompt` | `std_msgs/msg/String` | Subscribe / Publish | navigation_node(sub) / place_prompt_node(pub) | 言語指示 |
| `/cmd_vel` | `geometry_msgs/msg/Twist` | Publish | navigation_node | 速度指令 |
| `/toponav/current_node` | `std_msgs/msg/Int32` | Publish | place_prompt_node | 自己位置推定した現在ノードID |

## Findings

- **Action Expertはランダム初期化の方が良い(実機検証済み, 2026-07-09)**: SmolVLA公式重みの
  action expertはマニピュレータの物体把持タスクで事前学習されており、車輪ロボットのナビゲーションに
  finetuneすると負の転移が起きる。train lossだけでは差が出ないが、実機で比較するとランダム初期化した方が
  明らかに想定通りのカーブで曲がるようになった。VLM(視覚+言語理解)側の事前学習は維持したまま、
  action expertだけランダム初期化するのが現在のデフォルトレシピ。
- **多様なタスクを1モデルに詰め込むとgrad_normが一時的に再上昇することがある**: 単一ルートの反復学習では
  滑らかに収束するgrad_normが、多様な走行パターン+視覚拡張+数百種類のタスク文言を混ぜた学習では中盤で
  再上昇し、その後また下降する挙動が見られた。lossは横ばいのまま。視覚だけの近道を潰して言語指示を
  実際に使わせる学習ダイナミクスの可能性があるが、学習曲線だけでは実際に言語条件付けが機能しているか
  判断できず、実機/オフライン評価が必要。

## ライセンス

本リポジトリの独自実装部分は MIT License を想定しています。
一方で `lerobot/` はサブモジュールとして管理される別プロジェクトであり、`lerobot` 側のライセンスに従います。

- 本リポジトリ独自コード: MIT
- `lerobot/`: [lerobot/LICENSE](https://github.com/open-rdc/lerobot/blob/main/LICENSE) に従う
