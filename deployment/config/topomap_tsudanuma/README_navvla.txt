津田沼(魚眼)のトポロジカルマップ  2026-10-23 作成
- 元の走行: tsudanuma_20261013_123146(コースを一周した走行、スタートとゴールがほぼ同じ地点、9,427 フレーム、約 1.73 km)
  ※学習に使った周回(学習用 15 周の 1 つ)。本番の屋外実験の前に、学習に使っていない走行を録って作り直す予定
  ※以前のもの(評価用周回 134655 から作った版、コースの途中から始まるので不採用)は ..\tsudanuma_toponav_134655
- 作り方: NavVLA exp/tsukuba-1017 の deployment/scripts/create_topomap.py <データ> --saved-step 10 --crop-size 0
  (10 フレームおき = 943 ノード、魚眼は切り出さない、特徴は PlaceNet deployment/weights/placenet.pt)
- ノードの指示文: その周回の traj_prompt.txt の同じフレームの行(2026-10-20 に見直した注釈。v3 でも変更なし。指示なしのノードは 0)
- 実機では NavVLA の data/ に置く(nav.yaml の topomap_path / topomap_image_dir を tsudanuma_toponav に、toponav_crop_size: 0)
  SmolVLA の place_prompt_node で使うときも toponav_crop_size を 0 にすること(既定は 288 で中央を切り出す)
