"""Inspect a single Fashion-MNIST image through the ViT step by step.

This script is designed for debugging and for explaining tensor shapes in VS Code.
"""

from __future__ import annotations

import argparse
import gzip
import random
import struct
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from vit_model import VisionTransformer


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Debug one ViT forward pass")
	parser.add_argument("--data-root", type=str, default="data/FashionMNIST")
	parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
	parser.add_argument("--checkpoint-prefix", type=str, default="vit_fashionmnist")
	parser.add_argument("--checkpoint", type=str, default=None)
	parser.add_argument("--dataset", "--dataset-split", dest="dataset", type=str, choices=["train", "test"], default="test")
	parser.add_argument("--index", type=int, default=None)
	parser.add_argument("--no-show", action="store_true", help="Skip matplotlib image display during debugging")
	return parser.parse_args()


def _open_idx_file(path: Path):
	return gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb")


def read_idx_images(path: Path) -> torch.Tensor:
	with _open_idx_file(path) as file_obj:
		magic, num_images, rows, cols = struct.unpack(">IIII", file_obj.read(16))
		if magic != 2051:
			raise ValueError(f"Unexpected image file magic number: {magic}")
		buffer = file_obj.read()
	images = np.frombuffer(buffer, dtype=np.uint8).reshape(num_images, rows, cols).copy()
	return torch.from_numpy(images).float().div(255.0)


def read_idx_labels(path: Path) -> torch.Tensor:
	with _open_idx_file(path) as file_obj:
		magic, num_labels = struct.unpack(">II", file_obj.read(8))
		if magic != 2049:
			raise ValueError(f"Unexpected label file magic number: {magic}")
		buffer = file_obj.read()
	labels = np.frombuffer(buffer, dtype=np.uint8).copy()
	if len(labels) != num_labels:
		raise ValueError("Label file size does not match header")
	return torch.from_numpy(labels).long()


def find_latest_checkpoint(checkpoint_dir: Path, checkpoint_prefix: str) -> Path:
	checkpoint_dir = Path(checkpoint_dir)
	pattern = f"{checkpoint_prefix}_*.pth"
	candidates = sorted(checkpoint_dir.glob(pattern), key=lambda path: path.stat().st_mtime)
	if candidates:
		return candidates[-1]

	fallback = checkpoint_dir / f"{checkpoint_prefix}.pth"
	if fallback.exists():
		return fallback

	raise FileNotFoundError(
		f"No checkpoint found in {checkpoint_dir}. Expected files matching {pattern} or {fallback.name}."
	)


class LocalFashionMNIST(Dataset):
	def __init__(self, split_dir: Path, split: str) -> None:
		split_dir = Path(split_dir)
		if split == "train":
			image_path = split_dir / "train-images-idx3-ubyte.gz"
			label_path = split_dir / "train-labels-idx1-ubyte.gz"
		elif split == "test":
			image_path = split_dir / "t10k-images-idx3-ubyte.gz"
			label_path = split_dir / "t10k-labels-idx1-ubyte.gz"
		else:
			raise ValueError("split must be 'train' or 'test'")

		if not image_path.exists() or not label_path.exists():
			raise FileNotFoundError(
				f"Missing Fashion-MNIST files in {split_dir}. Expected {image_path.name} and {label_path.name}."
			)

		self.images = read_idx_images(image_path)
		self.labels = read_idx_labels(label_path)

		if len(self.images) != len(self.labels):
			raise ValueError("Image count does not match label count")

	def __len__(self) -> int:
		return len(self.labels)

	def __getitem__(self, index: int):
		image = self.images[index].unsqueeze(0)
		label = self.labels[index]
		return image, label


def load_model(checkpoint_path: Path, device: torch.device) -> VisionTransformer:
	model = VisionTransformer()
	model.to(device)

	if checkpoint_path.exists():
		checkpoint = torch.load(checkpoint_path, map_location=device)
		state_dict = checkpoint.get("model_state_dict", checkpoint)
		model.load_state_dict(state_dict)
		print(f"Loaded checkpoint: {checkpoint_path}")
	else:
		print(f"Checkpoint not found, using randomly initialized model: {checkpoint_path}")

	model.eval()
	return model


def load_sample(data_root: str, dataset_split: str, index: int | None):
	split_dir = Path(data_root) / dataset_split
	dataset = LocalFashionMNIST(split_dir, split=dataset_split)
	if index is None:
		index = random.randrange(len(dataset))
	image, label = dataset[index]
	return image.unsqueeze(0), label, index


def show_sample_image(image: torch.Tensor, index: int, label: int, prediction: int | None = None) -> None:
	import matplotlib.pyplot as plt

	image_2d = image.squeeze(0).squeeze(0).cpu().numpy()
	plt.figure(figsize=(4, 4))
	plt.imshow(image_2d, cmap="gray")
	if prediction is None:
		title = f"Fashion-MNIST | index={index} | label={label}"
	else:
		title = f"Fashion-MNIST | index={index} | label={label} | prediction={prediction}"
	plt.title(title)
	plt.text(
		0.02,
		0.96,
		f"label: {label}\nprediction: {prediction if prediction is not None else 'N/A'}",
		transform=plt.gca().transAxes,
		fontsize=10,
		color="yellow",
		verticalalignment="top",
		bbox=dict(facecolor="black", alpha=0.55, pad=4),
	)
	plt.axis("off")
	plt.tight_layout()
	plt.show()


def describe_forward(model: VisionTransformer, image: torch.Tensor) -> torch.Tensor:
	print(f"input          : {image.shape}")
	x = model.patch_embed(image)
	print(f"patch_embed    : {x.shape}")

	batch_size = x.shape[0]
	cls_token = model.cls_token.expand(batch_size, -1, -1)
	print(f"cls_token      : {cls_token.shape}")
	x = torch.cat((cls_token, x), dim=1)
	print(f"after concat   : {x.shape}")
	x = x + model.pos_embed
	print(f"after pos_embed: {x.shape}")
	x = model.pos_drop(x)

	for block_index, block in enumerate(model.blocks, start=1):
		x = block(x)
		print(f"block {block_index:<2}      : {x.shape}")

	x = model.norm(x)
	print(f"after norm     : {x.shape}")
	cls_output = x[:, 0]
	print(f"cls_output     : {cls_output.shape}")
	logits = model.head(cls_output)
	print(f"logits         : {logits.shape}")
	return logits


def main() -> None:
	args = parse_args()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	checkpoint_path = Path(args.checkpoint) if args.checkpoint is not None else find_latest_checkpoint(
		Path(args.checkpoint_dir), args.checkpoint_prefix
	)
	model = load_model(checkpoint_path, device)
	image, label, sample_index = load_sample(args.data_root, args.dataset, args.index)
	image = image.to(device)

	print(f"sample index    : {sample_index}")
	print(f"dataset split   : {args.dataset}")
	print(f"true label      : {label}")
	with torch.no_grad():
		logits = describe_forward(model, image)
		prediction = logits.argmax(dim=1).item()
		print(f"prediction      : {prediction}")
	if not args.no_show:
		show_sample_image(image, sample_index, int(label.item()), prediction)


if __name__ == "__main__":
	main()
