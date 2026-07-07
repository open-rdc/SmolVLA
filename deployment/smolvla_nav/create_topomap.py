#!/usr/bin/env python3
"""navvla_tsudanuma_nav6 (episode01..NN を番号順に連結した1周分のルート) から
言語指示付きのトポロジカルマップを生成する。

NavVLA の deployment/scripts/create_topomap.py がベース。差分:
  - トラジェクトリ(episodeNN)を数値として自然順ソートして読み込む
  - 各ノードに、対応フレームの traj_prompt.txt の行を `instruction` として付与する
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
import torch
import yaml
from torchvision import transforms


REPO_ROOT = Path(__file__).resolve().parents[2]


class TopomapGenerator:
    def __init__(
        self,
        dataset_path: Path,
        output_dir: Path,
        weight_path: Path,
        device: torch.device,
        saved_step: int = 10,
        crop_size: int = 288,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.output_image_dir = self.output_dir / "images"
        self.topomap_path = self.output_dir / "topomap.yaml"
        self.weight_path = Path(weight_path)
        self.device = device
        self.saved_step = int(saved_step)
        self.crop_size = int(crop_size)
        if self.saved_step < 1:
            raise ValueError("saved_step must be >= 1")

        self.model = torch.jit.load(str(self.weight_path), map_location=self.device).eval()
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((85, 85), antialias=True),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @staticmethod
    def _image_sort_key(path: Path) -> tuple[int, str]:
        try:
            return int(path.stem), path.name
        except ValueError:
            return 0, path.name

    @staticmethod
    def _natural_sort_key(name: str) -> tuple:
        # "episode01" -> ("episode", 1)  同名prefix内は数値順、それ以外は文字列順
        parts = re.split(r"(\d+)", name)
        return tuple(int(p) if p.isdigit() else p for p in parts)

    def _load_trajectory_names(self) -> list[str]:
        traj_names_path = self.dataset_path / "traj_names.txt"
        if not traj_names_path.exists():
            names = [path.name for path in self.dataset_path.glob("episode*") if path.is_dir()]
            if not names:
                names = [path.name for path in self.dataset_path.glob("traj_*") if path.is_dir()]
            return sorted(names, key=self._natural_sort_key)

        with traj_names_path.open("r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def _load_trajectory(self, traj_name: str) -> list[Path]:
        traj_dir = self.dataset_path / traj_name
        if not traj_dir.is_dir():
            raise FileNotFoundError(f"Trajectory directory not found: {traj_dir}")

        image_paths = sorted(
            list(traj_dir.glob("*.jpg")) + list(traj_dir.glob("*.png")),
            key=self._image_sort_key,
        )
        if not image_paths:
            raise ValueError(f"No images found in trajectory: {traj_dir}")

        return image_paths

    def _load_prompts(self, traj_name: str) -> list[str]:
        prompt_path = self.dataset_path / traj_name / "traj_prompt.txt"
        if not prompt_path.exists():
            raise FileNotFoundError(f"traj_prompt.txt not found: {prompt_path}")
        with prompt_path.open("r", encoding="utf-8") as f:
            return [line.rstrip("\n") for line in f]

    def _preprocess_image(self, image_path: Path) -> np.ndarray:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")

        height, width = image.shape[:2]
        side = min(height, width, self.crop_size)
        top = (height - side) // 2
        left = (width - side) // 2
        return image[top : top + side, left : left + side]

    def _extract_feature(self, image_bgr: np.ndarray) -> np.ndarray:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device, dtype=torch.float32)
        with torch.no_grad():
            feature = self.model(image_tensor)

        feature_np = feature.squeeze(0).detach().cpu().numpy().reshape(-1).astype(np.float32)
        norm = float(np.linalg.norm(feature_np))
        if norm > 1e-8:
            feature_np = feature_np / norm
        return feature_np

    def generate(self) -> Path:
        self.output_image_dir.mkdir(parents=True, exist_ok=True)
        traj_names = self._load_trajectory_names()
        if not traj_names:
            raise ValueError(f"No trajectories found in dataset: {self.dataset_path}")

        nodes = []
        features = []
        for traj_name in traj_names:
            image_paths = self._load_trajectory(traj_name)
            prompts = self._load_prompts(traj_name)
            if len(prompts) != len(image_paths):
                raise ValueError(
                    f"traj_prompt.txt line count ({len(prompts)}) != image count "
                    f"({len(image_paths)}) in trajectory: {traj_name}"
                )

            for list_index, image_path in enumerate(image_paths[:: self.saved_step]):
                node_index = len(nodes)
                frame_index = self._image_sort_key(image_path)[0]
                # image_paths はフレーム番号順ソート済みなので、間引き後も元のリスト順indexで引ける
                prompt_index = list_index * self.saved_step
                instruction = prompts[prompt_index]

                cropped_image = self._preprocess_image(image_path)

                save_image = cv2.resize(cropped_image, (85, 85), interpolation=cv2.INTER_AREA)
                output_image_name = f"img{node_index + 1:05d}.png"
                cv2.imwrite(str(self.output_image_dir / output_image_name), save_image)

                features.append(self._extract_feature(cropped_image))
                node = {
                    "id": node_index,
                    "image": output_image_name,
                    "instruction": instruction,
                    "source": {
                        "trajectory": traj_name,
                        "frame": frame_index,
                        "image": str(image_path.relative_to(self.dataset_path)),
                    },
                }
                nodes.append(node)

        for node_index, node in enumerate(nodes):
            target = node_index + 1 if node_index + 1 < len(nodes) else node_index
            node["edges"] = [{"target": target}]

        # 特徴量(512次元 x ノード数)はYAMLに埋め込むとパースが極端に遅くなる(926ノードで20秒超)ため、
        # 別ファイル(.npy)に保存し、topomap.yaml からは features_path で参照する。
        features_array = np.stack(features, axis=0).astype(np.float32)
        features_path = self.output_dir / "topomap_features.npy"
        np.save(features_path, features_array)

        with self.topomap_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(
                {"features_path": features_path.name, "nodes": nodes},
                f,
                sort_keys=False,
                allow_unicode=True,
            )
        return self.topomap_path


def resolve_cli_path(raw_path: str, base_path: Path = Path.cwd()) -> Path:
    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else base_path / path


def main(args: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_path",
        nargs="?",
        default="deployment/data/navvla_tsudanuma_nav6",
        help="episodeNN ディレクトリ群を含むデータセットディレクトリ（相対パスはSmolVLAリポジトリルート基準）",
    )
    parser.add_argument(
        "--output-dir",
        default="deployment/config/topomap",
        help="topomap.yaml と images/ の出力先（SmolVLAリポジトリルート基準）",
    )
    parser.add_argument(
        "--weights",
        default="deployment/weights/placenet.pt",
        help="PlaceNet TorchScript 重みパス（SmolVLAリポジトリルート基準）",
    )
    parser.add_argument("--saved-step", type=int, default=10, help="何フレームおきにノード化するか")
    parser.add_argument("--crop-size", type=int, default=288, help="85x85にリサイズする前の中央クロップサイズ")
    parsed = parser.parse_args(args)

    dataset_path = resolve_cli_path(parsed.dataset_path, REPO_ROOT)
    output_dir = resolve_cli_path(parsed.output_dir, REPO_ROOT)
    weights_path = resolve_cli_path(parsed.weights, REPO_ROOT)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = TopomapGenerator(
        dataset_path=dataset_path,
        output_dir=output_dir,
        weight_path=weights_path,
        device=device,
        saved_step=parsed.saved_step,
        crop_size=parsed.crop_size,
    )
    topomap_path = generator.generate()
    print(f"Topomap saved: {topomap_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
