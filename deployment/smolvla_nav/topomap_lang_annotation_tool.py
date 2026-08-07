#!/usr/bin/env python3
"""トポロジカルマップ (topomap.yaml) 上で言語指示を調整する GUI ツール。

NavVLA の ``tools/lang_anotation_tool.py``（1エピソード内をスライダーで範囲選択し
``traj_prompt.txt`` に言語指示を書き込むツール）がベース。差分:

  - 対象が「1エピソードのフレーム列」ではなく「create_topomap.py が生成した
    topomap.yaml のノード列（複数エピソードを連結したトポロジカルマップ）」になる。
    フレームインデックスの代わりにノードインデックス（グラフ順、= id 順）で
    Start/End スライダーを操作する。
  - 全ノードを俯瞰できるマップビュー（instruction ごとに色分けした帯）を追加。
    帯をクリックするとそのノードへジャンプする。
  - エピソード切り替えの代わりに「Prev/Next Segment」で instruction が変化する
    境界へジャンプできる（隣接ノードで同じ instruction が続く区間 = セグメント）。
  - 保存は traj_prompt.txt への行書き込みではなく、選択範囲のノードの
    ``instruction`` フィールドを書き換えて topomap.yaml を再ダンプする。
    place_prompt_node.py はこの ``instruction`` を自己位置推定結果に応じて
    そのまま /prompt (SmolVLA への言語指示) として配信するため、この編集結果は
    そのまま学習データにもデプロイ時のナビゲーションにも反映される。
  - 初回保存時に topomap.yaml.bak を作成する（誤編集からの復旧用）。

Usage:
    python3 topomap_lang_annotation_tool.py [topomap_dir]

``topomap_dir`` は ``topomap.yaml`` と ``images/`` を含むディレクトリ
（省略時は ``deployment/config/topomap``、SmolVLA リポジトリルート基準）。
"""

from __future__ import annotations

import argparse
import colorsys
import hashlib
import shutil
import tkinter as tk
from pathlib import Path
from tkinter import messagebox
from typing import Optional

import yaml
from PIL import Image, ImageTk

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INSTRUCTION = "No language instruction"
DISPLAY_IMAGE_SIZE: tuple[int, int] = (224, 224)

OVERVIEW_CANVAS_WIDTH = 900
OVERVIEW_ROW_HEIGHT = 22
OVERVIEW_NODES_PER_ROW = 100
OVERVIEW_VISIBLE_ROWS = 6


def _color_for_instruction(text: str) -> str:
    """instruction 文字列から決定的に色を割り当てる（同じ文言は常に同じ色）。"""
    if not text or text == DEFAULT_INSTRUCTION:
        return "#cccccc"
    digest = hashlib.md5(text.encode("utf-8")).digest()
    hue = digest[0] / 255.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.55, 0.92)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


class TopomapLangAnnotationTool:
    """トポロジカルマップの各ノードに紐づく言語指示を編集する GUI。"""

    def __init__(self, topomap_dir: Path) -> None:
        self.topomap_dir = Path(topomap_dir)
        self.topomap_path = self.topomap_dir / "topomap.yaml"
        self.image_dir = self.topomap_dir / "images"

        self.root = tk.Tk()
        self.root.title("Topomap Lang Annotation Tool")

        self.data: dict = {}
        self.nodes: list[dict] = []
        self.instructions: list[str] = []
        self.segments: list[tuple[int, int]] = []
        self.start_photo: Optional[ImageTk.PhotoImage] = None
        self.end_photo: Optional[ImageTk.PhotoImage] = None
        self._loaded_lang: str = ""
        self._node_boxes: list[tuple[int, int, int, int]] = []  # canvas item id -> (x0,y0,x1,y1) は不要、item id引きで十分
        self._backed_up = False

        self.start_var = tk.IntVar(value=0)
        self.end_var = tk.IntVar(value=0)
        self.lang_var = tk.StringVar()
        self.status_var = tk.StringVar()
        self.jump_var = tk.StringVar()

        self.load_topomap()
        self.init_gui()
        self.select_range(0, 0)

    # ------------------------------------------------------------------ #
    # データ読み込み / 保存
    # ------------------------------------------------------------------ #

    def load_topomap(self) -> None:
        if not self.topomap_path.exists():
            raise FileNotFoundError(f"topomap.yaml not found: {self.topomap_path}")
        with self.topomap_path.open("r", encoding="utf-8") as f:
            self.data = yaml.safe_load(f) or {}

        raw_nodes = self.data.get("nodes", [])
        if not raw_nodes:
            raise ValueError(f"No nodes found in topomap: {self.topomap_path}")

        # id順に並べる（create_topomap.py は既にid順だが、手編集されたファイルにも耐える）
        self.nodes = sorted(raw_nodes, key=lambda n: int(n.get("id", 0)))
        self.instructions = [str(n.get("instruction", "")) or DEFAULT_INSTRUCTION for n in self.nodes]
        self.recompute_segments()

    def recompute_segments(self) -> None:
        segments: list[tuple[int, int]] = []
        start = 0
        for i in range(1, len(self.instructions) + 1):
            if i == len(self.instructions) or self.instructions[i] != self.instructions[start]:
                segments.append((start, i - 1))
                start = i
        self.segments = segments

    def save_annotation(self) -> None:
        s, e = self.start_var.get(), self.end_var.get()
        lang = self.lang_var.get().strip() or DEFAULT_INSTRUCTION
        for pos in range(s, e + 1):
            self.instructions[pos] = lang
            self.nodes[pos]["instruction"] = lang

        self._backup_once()
        with self.topomap_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(self.data, f, sort_keys=False, allow_unicode=True)

        self._loaded_lang = lang
        self.recompute_segments()
        self.draw_overview()
        self.status_var.set(f"Saved: nodes [{s}→{e}] ({e - s + 1} nodes) ← '{lang}'")
        # 保存後は Lang 欄からフォーカスを外し、矢印キーによるノード送りをすぐ再開できるようにする。
        self._release_entry_focus()

    def _backup_once(self) -> None:
        if self._backed_up:
            return
        backup_path = self.topomap_path.with_suffix(self.topomap_path.suffix + ".bak")
        if not backup_path.exists():
            shutil.copy2(self.topomap_path, backup_path)
        self._backed_up = True

    # ------------------------------------------------------------------ #
    # GUI 構築
    # ------------------------------------------------------------------ #

    def init_gui(self) -> None:
        root = self.root

        header = tk.Label(
            root,
            text=f"{self.topomap_path}  ({len(self.nodes)} nodes, {len(self.segments)} segments)",
            anchor="w",
        )
        header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=8, pady=(6, 2))

        self._init_overview(row=1)
        self._init_segment_nav(row=2)
        self._init_image_panels(row=3, image_row=4, scale_row=5, lang_row=6)
        self._init_bottom(row=7)

        self.root.bind("<Return>", lambda _: self.save_annotation())
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Scale にフォーカスが無い場合（俯瞰マップ・ラベル・ウィンドウ全体など）のための
        # 矢印キー処理。Entry (Lang / Jump to node id) にフォーカスがある間はテキストの
        # カーソル移動を優先し、ノード選択の移動は行わない。
        self.root.bind("<Left>", lambda _e: self._handle_arrow(-1, "both"))
        self.root.bind("<Right>", lambda _e: self._handle_arrow(1, "both"))
        self.root.bind("<Shift-Left>", lambda _e: self._handle_arrow(-1, "end"))
        self.root.bind("<Shift-Right>", lambda _e: self._handle_arrow(1, "end"))
        self.root.bind("<Control-Left>", lambda _e: self._handle_arrow(-1, "start"))
        self.root.bind("<Control-Right>", lambda _e: self._handle_arrow(1, "start"))

    def _init_overview(self, row: int) -> None:
        wrapper = tk.Frame(self.root)
        wrapper.grid(row=row, column=0, columnspan=2, padx=8, pady=4, sticky="ew")

        n_rows = max(1, -(-len(self.nodes) // OVERVIEW_NODES_PER_ROW))
        visible_h = min(n_rows, OVERVIEW_VISIBLE_ROWS) * OVERVIEW_ROW_HEIGHT

        self.overview_canvas = tk.Canvas(
            wrapper, width=OVERVIEW_CANVAS_WIDTH, height=visible_h,
            bg="white", highlightthickness=1, highlightbackground="#999",
        )
        vscroll = tk.Scrollbar(wrapper, orient=tk.VERTICAL, command=self.overview_canvas.yview)
        self.overview_canvas.configure(yscrollcommand=vscroll.set)
        self.overview_canvas.grid(row=0, column=0, sticky="ew")
        vscroll.grid(row=0, column=1, sticky="ns")

        self.overview_canvas.bind("<Button-1>", self.on_overview_click)
        self.draw_overview()

    def _init_segment_nav(self, row: int) -> None:
        bar = tk.Frame(self.root)
        bar.grid(row=row, column=0, columnspan=2, pady=4)

        tk.Button(bar, text="⏮ Prev Segment", command=self.prev_segment).pack(side=tk.LEFT, padx=4)
        tk.Button(bar, text="Next Segment ⏭", command=self.next_segment).pack(side=tk.LEFT, padx=4)
        self.segment_label = tk.Label(bar, text="", width=18)
        self.segment_label.pack(side=tk.LEFT, padx=8)

        tk.Label(bar, text="Jump to node id:").pack(side=tk.LEFT, padx=(16, 2))
        self.jump_entry = tk.Entry(bar, textvariable=self.jump_var, width=8)
        self.jump_entry.pack(side=tk.LEFT)
        self.jump_entry.bind("<Escape>", self._release_entry_focus)
        tk.Button(bar, text="Go", command=self.jump_to_node).pack(side=tk.LEFT, padx=4)

        tk.Label(
            bar,
            text="(← → : ±1node  Shift+← →: resize end  Ctrl+← →: resize start)",
            fg="gray",
        ).pack(side=tk.LEFT, padx=(16, 2))

    def _init_image_panels(self, row: int, image_row: int, scale_row: int, lang_row: int) -> None:
        self.start_index_label = tk.Label(self.root, text="Start Index: 0")
        self.start_index_label.grid(row=row, column=0)
        self.end_index_label = tk.Label(self.root, text="End Index: 0")
        self.end_index_label.grid(row=row, column=1)

        self.start_panel = tk.Label(self.root)
        self.start_panel.grid(row=image_row, column=0, padx=8)
        self.end_panel = tk.Label(self.root)
        self.end_panel.grid(row=image_row, column=1, padx=8)

        n = max(0, len(self.nodes) - 1)
        self.start_scale = tk.Scale(
            self.root, variable=self.start_var, orient=tk.HORIZONTAL,
            from_=0, to=n, length=320, command=self.on_start_changed,
        )
        self.start_scale.grid(row=scale_row, column=0, padx=8)
        self.end_scale = tk.Scale(
            self.root, variable=self.end_var, orient=tk.HORIZONTAL,
            from_=0, to=n, length=320, command=self.on_end_changed,
        )
        self.end_scale.grid(row=scale_row, column=1, padx=8)

        # tk.Scale はフォーカスを持つと Left/Right を内蔵ハンドリングして±1動かすため、
        # そのままだとルートの矢印キー処理と二重に動いてしまう。ここで独自ハンドラに
        # 差し替え、"break" で伝播を止めて二重発火を防ぐ。
        for scale in (self.start_scale, self.end_scale):
            scale.bind("<Left>", lambda _e: self._consume_arrow(-1, "both"))
            scale.bind("<Right>", lambda _e: self._consume_arrow(1, "both"))
            scale.bind("<Shift-Left>", lambda _e: self._consume_arrow(-1, "end"))
            scale.bind("<Shift-Right>", lambda _e: self._consume_arrow(1, "end"))
            scale.bind("<Control-Left>", lambda _e: self._consume_arrow(-1, "start"))
            scale.bind("<Control-Right>", lambda _e: self._consume_arrow(1, "start"))

        self.start_lang_label = tk.Label(self.root, text="", fg="blue", wraplength=320, justify="left")
        self.start_lang_label.grid(row=lang_row, column=0, padx=8, sticky="w")
        self.end_lang_label = tk.Label(self.root, text="", fg="blue", wraplength=320, justify="left")
        self.end_lang_label.grid(row=lang_row, column=1, padx=8, sticky="w")

    def _init_bottom(self, row: int) -> None:
        bottom = tk.Frame(self.root)
        bottom.grid(row=row, column=0, columnspan=2, pady=4, sticky="ew")

        self.range_label = tk.Label(bottom, text="")
        self.range_label.grid(row=0, column=0, columnspan=3, sticky="w", padx=8)
        self.source_label = tk.Label(bottom, text="", fg="gray")
        self.source_label.grid(row=1, column=0, columnspan=3, sticky="w", padx=8)

        tk.Label(bottom, text="Lang:").grid(row=2, column=0, padx=4)
        self.lang_entry = tk.Entry(bottom, textvariable=self.lang_var, width=60)
        self.lang_entry.grid(row=2, column=1, padx=4)
        self.lang_entry.bind("<Escape>", self._release_entry_focus)
        tk.Button(bottom, text="Save", command=self.save_annotation).grid(row=2, column=2, padx=8)

        tk.Label(bottom, textvariable=self.status_var, fg="gray").grid(
            row=3, column=0, columnspan=3, sticky="w", padx=8,
        )

    # ------------------------------------------------------------------ #
    # 俯瞰マップ描画
    # ------------------------------------------------------------------ #

    def draw_overview(self) -> None:
        canvas = self.overview_canvas
        canvas.delete("all")
        n = len(self.nodes)
        if n == 0:
            return
        cell_w = OVERVIEW_CANVAS_WIDTH / OVERVIEW_NODES_PER_ROW
        n_rows = -(-n // OVERVIEW_NODES_PER_ROW)

        for pos in range(n):
            row = pos // OVERVIEW_NODES_PER_ROW
            col = pos % OVERVIEW_NODES_PER_ROW
            x0, y0 = col * cell_w, row * OVERVIEW_ROW_HEIGHT
            x1, y1 = x0 + cell_w, y0 + OVERVIEW_ROW_HEIGHT
            color = _color_for_instruction(self.instructions[pos])
            canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="", tags=(f"node{pos}", "node"))

        canvas.configure(scrollregion=(0, 0, OVERVIEW_CANVAS_WIDTH, n_rows * OVERVIEW_ROW_HEIGHT))
        self.draw_overview_selection()

    def draw_overview_selection(self) -> None:
        canvas = self.overview_canvas
        canvas.delete("selection")
        s, e = self.start_var.get(), self.end_var.get()
        cell_w = OVERVIEW_CANVAS_WIDTH / OVERVIEW_NODES_PER_ROW
        for pos in range(s, e + 1):
            row = pos // OVERVIEW_NODES_PER_ROW
            col = pos % OVERVIEW_NODES_PER_ROW
            x0, y0 = col * cell_w, row * OVERVIEW_ROW_HEIGHT
            x1, y1 = x0 + cell_w, y0 + OVERVIEW_ROW_HEIGHT
            canvas.create_rectangle(x0, y0, x1, y1, outline="black", width=2, tags="selection")

    def on_overview_click(self, event: tk.Event) -> None:
        canvas_y = self.overview_canvas.canvasy(event.y)
        row = int(canvas_y // OVERVIEW_ROW_HEIGHT)
        col = int(event.x // (OVERVIEW_CANVAS_WIDTH / OVERVIEW_NODES_PER_ROW))
        pos = row * OVERVIEW_NODES_PER_ROW + col
        if 0 <= pos < len(self.nodes):
            self.confirm_and_select(pos, pos)

    # ------------------------------------------------------------------ #
    # 選択 / ナビゲーション
    # ------------------------------------------------------------------ #

    def select_range(self, s: int, e: int) -> None:
        n = len(self.nodes)
        s = max(0, min(s, n - 1))
        e = max(s, min(e, n - 1))
        self.start_var.set(s)
        self.end_var.set(e)
        lang = self.instructions[s]
        self.lang_var.set(lang)
        self._loaded_lang = lang
        self.update_display()

    def has_unsaved_changes(self) -> bool:
        return self.lang_var.get() != self._loaded_lang

    def confirm_and_select(self, s: int, e: int) -> None:
        if self.has_unsaved_changes():
            if not messagebox.askyesno("Unsaved changes", "Discard the current edit and move on?"):
                return
        self.select_range(s, e)

    def on_close(self) -> None:
        """ウィンドウを閉じる操作（Xボタン等）。未保存の Lang 編集があれば確認する。

        Save 済みの変更は Save のたびに毎回 topomap.yaml へ書き込み済みなので、
        ここで警告が出なければそのまま閉じて問題ない。
        """
        if self.has_unsaved_changes():
            if not messagebox.askyesno(
                "Unsaved changes",
                "The current Lang edit has not been saved. Close without saving?",
            ):
                return
        self.root.destroy()

    def _consume_arrow(self, delta: int, target: str) -> str:
        """Scale ウィジェット用ハンドラ。内蔵の Left/Right 移動を止めて自前の1ノード刻みに置き換える。"""
        self._handle_arrow(delta, target)
        return "break"

    def _handle_arrow(self, delta: int, target: str) -> None:
        if isinstance(self.root.focus_get(), tk.Entry):
            return
        n = len(self.nodes)
        s, e = self.start_var.get(), self.end_var.get()
        if target == "start":
            new_s, new_e = max(0, min(s + delta, e)), e
        elif target == "end":
            new_s, new_e = s, max(s, min(e + delta, n - 1))
        else:  # "both": 幅を保ったまま選択範囲そのものを1ノードずつ動かす
            width = e - s
            new_s = max(0, min(s + delta, n - 1 - width))
            new_e = new_s + width
        if (new_s, new_e) != (s, e):
            self.confirm_and_select(new_s, new_e)

    def _release_entry_focus(self, _event: Optional[tk.Event] = None) -> None:
        """Lang / Jump 欄からフォーカスを外し、矢印キーをノード送りに戻す（Esc、Save後に使用）。"""
        self.overview_canvas.focus_set()

    def jump_to_node(self) -> None:
        raw = self.jump_var.get().strip()
        if not raw:
            return
        try:
            node_id = int(raw)
        except ValueError:
            self.status_var.set(f"Invalid node id: {raw!r}")
            return
        pos = self._position_for_id(node_id)
        if pos is None:
            self.status_var.set(f"Node id not found: {node_id}")
            return
        self.confirm_and_select(pos, pos)

    def _position_for_id(self, node_id: int) -> Optional[int]:
        for pos, node in enumerate(self.nodes):
            if int(node.get("id", pos)) == node_id:
                return pos
        return None

    def current_segment_index(self) -> int:
        pos = self.start_var.get()
        for idx, (s, e) in enumerate(self.segments):
            if s <= pos <= e:
                return idx
        return 0

    def prev_segment(self) -> None:
        idx = self.current_segment_index()
        if idx == 0:
            self.status_var.set("Already at the first segment")
            return
        s, e = self.segments[idx - 1]
        self.confirm_and_select(s, e)

    def next_segment(self) -> None:
        idx = self.current_segment_index()
        if idx >= len(self.segments) - 1:
            self.status_var.set("Already at the last segment")
            return
        s, e = self.segments[idx + 1]
        self.confirm_and_select(s, e)

    def on_start_changed(self, value: str) -> None:
        s = int(value)
        if s > self.end_var.get():
            self.end_var.set(s)
        self.update_display()

    def on_end_changed(self, value: str) -> None:
        e = int(value)
        if e < self.start_var.get():
            self.start_var.set(e)
        self.update_display()

    # ------------------------------------------------------------------ #
    # 表示更新
    # ------------------------------------------------------------------ #

    def update_display(self) -> None:
        self.update_image_panel("start", self.start_var.get())
        self.update_image_panel("end", self.end_var.get())
        self.update_range_label()
        self.draw_overview_selection()
        s, e = self.start_var.get(), self.end_var.get()
        self.start_lang_label.configure(text=self.instructions[s])
        self.end_lang_label.configure(text=self.instructions[e])
        self.segment_label.configure(
            text=f"Segment {self.current_segment_index() + 1} / {len(self.segments)}"
        )

    def update_image_panel(self, side: str, pos: int) -> None:
        node = self.nodes[pos]
        image_path = self.image_dir / str(node["image"])
        img = Image.open(image_path).resize(DISPLAY_IMAGE_SIZE, Image.NEAREST)
        photo = ImageTk.PhotoImage(img)
        if side == "start":
            self.start_panel.configure(image=photo)
            self.start_photo = photo
            self.start_index_label.configure(text=f"Start Index: {node.get('id', pos)}")
        else:
            self.end_panel.configure(image=photo)
            self.end_photo = photo
            self.end_index_label.configure(text=f"End Index: {node.get('id', pos)}")

    def update_range_label(self) -> None:
        s, e = self.start_var.get(), self.end_var.get()
        self.range_label.configure(text=f"Range: [{s}] ─ [{e}]  ({e - s + 1} nodes)")
        sources = []
        for pos in (s, e):
            src = self.nodes[pos].get("source", {})
            traj = src.get("trajectory", "?")
            frame = src.get("frame", "?")
            sources.append(f"{traj} / frame {frame}")
        self.source_label.configure(text=f"source: [{sources[0]}] ─ [{sources[1]}]")

    def run(self) -> None:
        self.root.mainloop()


def resolve_cli_path(raw_path: str, base_path: Path = Path.cwd()) -> Path:
    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else base_path / path


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "topomap_dir",
        nargs="?",
        default="deployment/config/topomap",
        help="topomap.yaml と images/ を含むディレクトリ（相対パスはSmolVLAリポジトリルート基準）",
    )
    parsed = parser.parse_args(argv)
    topomap_dir = resolve_cli_path(parsed.topomap_dir, REPO_ROOT)
    TopomapLangAnnotationTool(topomap_dir).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
