# SmolVLA

## Overview

SmolVLA は、HuggingFace [`lerobot`](https://github.com/huggingface/lerobot) の
[SmolVLA](https://huggingface.co/lerobot/smolvla_base) を用いた、差動2輪移動ロボット向けナビゲーション実装です。
本リポジトリには、以下の 2 つの機能が含まれます。

- `SmolVLA` のファインチューニング(移動ロボットのナビゲーション向けに action/state を再定義)
- ROS 2 ノードによるナビゲーション推論

`lerobot` 本体はサブモジュールとして管理され、本リポジトリ側ではデータセット変換、学習ジョブ、
可視化スクリプト、ROS 2 ノードを提供します。

## Requirements

| 項目 | 内容 |
|---|---|
| OS | Ubuntu 22.04(ROS 2 Humble前提) |
| Python | 3.12(学習/lerobot) / 3.10(ROS 2 Humble) |
| ROS 2 | Humble |
| GPU | 学習は要CUDA GPU(目安 VRAM 20GB, `batch_size=8` で確認済み)。推論のみなら軽量でCPUでも動作 |

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

3つのノードで構成されます。**推論と操舵は別ノードに分かれています**(予測経路が `/cmd_vel` に
正しく反映されていなかった不具合の修正で分離しました)。

| ノード | 役割 |
|---|---|
| `navigation_node` | 画像と言語指示を購読し、`SmolVLAPolicy` で行動チャンクを推論。予測経路 `/smolvla_pred_path` とフォールバック用の生の速度指令 `/smolvla_cmd_vel_raw` を publish する。**`/cmd_vel` は出さない** |
| `path_follower_node` | `/smolvla_pred_path` を Pure Pursuit で追従し、最終的な `/cmd_vel` を publish する |
| `place_prompt_node` | 走行データから作ったトポロジカルマップ上で自己位置推定を行い、現在位置に対応する言語指示を `/prompt` に自動配信する |

推論は receding horizon で、行動チャンクの先頭数stepだけを実行して次tickで撮り直します。

ROS 2(Humble, Python 3.10)と `lerobot`(Python 3.12)はプロセスを分けず、同一プロセス内で直接importする構成です。

行動生成はフローマッチングで、ODE `dx/dt = v(x, t, obs)` を **Heun法(2段2次)** で t=1→0 へ積分します。
1ステップにつき速度場を2回評価するため、**推論時間は `2 × num_steps` に比例**します
(既定 `num_steps=4` で評価8回)。詳しくは [Findings](#findings) を参照。

起動ファイル:

- [`deployment/launch/smolvla_nav.launch.py`](https://github.com/open-rdc/SmolVLA/blob/main/deployment/launch/smolvla_nav.launch.py)

### 起動方法

```bash
ros2 launch smolvla_nav smolvla_nav.launch.py                    # トポロジカルマップで自己位置推定→自動プロンプト切替
ros2 launch smolvla_nav smolvla_nav.launch.py use_toponav:=false # 固定プロンプトのみ(place_prompt_nodeを止める)
```

主な launch 引数:

| 引数 | 既定 | 説明 |
|---|---|---|
| `use_toponav` | `true` | `false` で `place_prompt_node` を止め、固定プロンプトのみで動かす |
| `num_steps` | `0` | デノイズのステップ数。`0` はポリシー側の既定(`4`)。**推論時間は `2 × num_steps` に比例**。`2` 以下に下げるとステップ幅が大きすぎて精度が落ちる |
| `use_pure_pursuit` | `true` | `false` で `navigation_node` の生 dyaw をそのまま操舵に使う |
| `lookahead_distance` | `2.5` | Pure Pursuit の前方注視距離[m]。計画の不感帯(約1〜2m)より長く取ること |
| `path_timeout_sec` | `5.0` | `/smolvla_pred_path` がこれより古ければ生 dyaw にフォールバックする[s]。推論レイテンシより大きくすること |
| `step_lookahead` | `0` | chunk の何ステップ先の行動をフォールバック操舵に使うか(不感帯の実測用) |

トポロジカルマップの作成:

```bash
ros2 run smolvla_nav create_topomap --ros-args -p data_dir:=<走行データ> -p output:=deployment/config/topomap
```

### Topic 一覧

| Topic | 型 | 方向 | Node | 内容 |
|---|---|---|---|---|
| `/image_raw` | `sensor_msgs/msg/Image` | Subscribe | navigation_node, place_prompt_node | 現在観測画像 |
| `/autonomous` | `std_msgs/msg/Bool` | Subscribe | navigation_node, path_follower_node | 自律動作の有効/無効 |
| `/prompt` | `std_msgs/msg/String` | Subscribe / Publish | navigation_node(sub) / place_prompt_node(pub) | 言語指示 |
| `/smolvla_pred_path` | `nav_msgs/msg/Path` | Publish / Subscribe | navigation_node(pub) / path_follower_node(sub) | 予測した行動チャンクを base_link 基準の経路に積分したもの |
| `/smolvla_cmd_vel_raw` | `geometry_msgs/msg/Twist` | Publish / Subscribe | navigation_node(pub) / path_follower_node(sub) | モデルの生の速度指令。Pure Pursuit 無効時・経路が古いときのフォールバック |
| `/cmd_vel` | `geometry_msgs/msg/Twist` | Publish | **path_follower_node** | 最終的な速度指令 |
| `/smolvla_lookahead` | `geometry_msgs/msg/PointStamped` | Publish | path_follower_node | Pure Pursuit の注視点(調整用。`/smolvla_pred_path` と重ねて見る) |
| `/toponav/current_node` | `std_msgs/msg/Int32` | Publish | place_prompt_node | 自己位置推定した現在ノードID |

## Findings

- **フローマッチングの積分を Heun法(2次)にすると、速度と精度が同時に改善する(2026-08-11)**:
  推論は ODE `dx/dt = v(x, t, obs)` を t=1→0 へ積分する処理で、上流実装は陽的オイラー法
  (1段1次、`num_steps=10` で速度場の評価10回)だった。**推論時間はステップ数ではなく
  「速度場の評価回数」に比例する**ので、比較は評価回数を揃えて行う必要がある。
  実データ32フレーム×3ノイズでノイズを固定し、参照解 Heun N=100 との dyaw 誤差を測ると:

  | 評価回数 | Euler | Heun | 誤差比 |
  |---:|---|---|---|
  | 6 | N=6 0.1144 | N=3 0.0809 | Heun が 1.4分の1 |
  | 8 | N=8 0.0876 | **N=4 0.0469** | Heun が 1.9分の1 |
  | 10 | N=10 0.0733 | N=5 0.0331 | Heun が 2.2分の1 |

  収束次数の実測は Euler 0.86〜0.93 / Heun 1.73〜1.83 で、理論値(1 / 2)の85〜93%。
  **旧構成の Euler N=10(評価10回)に対し Heun N=4 は評価8回で誤差が約0.6倍**になり、
  速度と精度の両方で勝つため既定にした。ただし**評価回数を減らせば誤差自体は増える**点に注意
  (Heun N=3 は評価6回まで減るが誤差は旧構成の1.1倍)。評価4回以下ではステップ幅が大きすぎて
  修正子が割に合わず、1次のオイラー法のほうが正確になる。
  なお ω のバイアスの向きも逆で、Euler は ω を小さく外す(縮み 0.92/0.89)のに対し Heun は
  大きく外す(1.27/1.07)。Euler の向きは実機で対処済みの「カーブで膨らむ」問題と同じ向き。
  fp16 の丸め誤差は離散化誤差の1〜6%で無視できる。
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

## LICENSE

本リポジトリの独自実装部分は MIT License を想定しています。
一方で `lerobot/` はサブモジュールとして管理される別プロジェクトであり、`lerobot` 側のライセンスに従います。

- 本リポジトリ独自コード: MIT
- `lerobot/`: [lerobot/LICENSE](https://github.com/open-rdc/lerobot/blob/main/LICENSE) に従う
