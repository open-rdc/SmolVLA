# 未知の言い換え指示に対する言語汎化の評価

issue [#6](https://github.com/open-rdc/SmolVLA/issues/6) の実験一式。
issue に載せた数値はすべて `scripts/analyze.py` で再現できる。

## 構成

```
experiments/language/
├── scripts/
│   ├── derive_cls.py    指示クラスを文字列推定ではなく実データのΨから決める
│   ├── gen_para2.py     言い換えの生成器。267指示 → L1/L2/L3 計1,846本
│   ├── extract_obs.py   フレーム選抜と観測の切り出し。GPU不要。出力 v3_obs.npz（258MB）
│   ├── v3_infer.py      推論。v3_obs.npz だけ読む（データセット不要）
│   └── analyze.py       解析。issue の数値をすべて再現する
└── data/
    ├── theta.json               Ψ→クラス判定の閾値 θ=0.090 rad / ホライズン N=25
    ├── v3_frames.json           評価フレーム 1,064
    ├── data_cls.json            指示667種のΨ分布とデータ由来クラス（derive_cls.py の出力）
    ├── sel_instructions_v3.json 選抜した267指示（左折89・右折89・直進89）
    ├── paraphrases_v3.json      言い換え 1,846本
    ├── meta_v3.json             各言い換えの検索置換先・埋め込み距離・TF-IDF距離
    ├── train_vocab_full.json    学習語彙 667種
    ├── train_words.json         語彙に出現する異なり単語 208語
    └── v3_results.json.gz       推論結果 75,000件
```

## 再現

### 解析だけやり直す（数分、GPU不要）

`data/` に結果まで入っているので、これだけで issue の全数値が出る。

```
python scripts/analyze.py --work data --boot 4000
```

`--boot` はクラスタブートストラップの反復数（0で省略）。`analyze.py` は
`v3_results.json` が無ければ `v3_results.json.gz` を読む。

### 推論からやり直す（2070 Max-Q で約2.9時間）

観測ファイル `v3_obs.npz`（258MB）はサイズの都合でリポジトリに入れていない。
LeRobot データセットから作り直す。

```
WORK=data SMOLVLA_DATA=../../training/data python scripts/extract_obs.py 4
WORK=data python scripts/v3_infer.py --bench 12 --bs 8 --num-steps 4   # 速度実測
WORK=data python scripts/v3_infer.py --bs 8 --num-steps 4              # 本番
python scripts/analyze.py --work data --boot 4000
```

### 言い換えとラベルから作り直す

```
WORK=data SMOLVLA_DATA=../../training/data python scripts/derive_cls.py   # → data_cls.json
WORK=data python scripts/gen_para2.py                                     # → paraphrases_v3.json
```

## 環境変数

| 変数 | 既定 | 意味 |
|---|---|---|
| `WORK` | `.` | 入出力ディレクトリ。上の例では `data` |
| `SMOLVLA_DATA` | `../../training/data` | LeRobot データセット（`orne_*_tc`）の置き場 |

## 同梱していないもの

| | 理由 | 入手 |
|---|---|---|
| `v3_obs.npz`（258MB） | GitHub の100MB制限 | `extract_obs.py` で再生成 |
| 学習済みチェックポイント | サイズ | `smolvla_rec5_v2` の `checkpoints/last/pretrained_model` |
| LeRobot データセット | サイズ | `orne_box_all_tc_rec5_v2`（4,685 ep / 340,899 frame） |

`v3_gpu.py`（GPGPU クラスタ向けの推論スクリプト）は共有ストレージの絶対パスが
埋まっているため同梱していない。`v3_infer.py` がローカル版で、機能は同じ。
