"""Train a small Vision Transformer on Fashion-MNIST.

This script is intentionally straightforward so it can be explained line by line.
"""

from __future__ import annotations

import argparse
import gzip
from datetime import datetime
import struct
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from vit_model import VisionTransformer


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train a small ViT on Fashion-MNIST")
	parser.add_argument("--data-root", type=str, default="data/FashionMNIST")
	parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
	parser.add_argument("--checkpoint-prefix", type=str, default="vit_fashionmnist")
	parser.add_argument("--resume-checkpoint", type=str, default=None)
	parser.add_argument("--batch-size", type=int, default=64)
	parser.add_argument("--epochs", type=int, default=5)
	parser.add_argument("--learning-rate", type=float, default=3e-4)
	parser.add_argument("--weight-decay", type=float, default=0.05)
	parser.add_argument("--val-split", type=float, default=0.1)
	parser.add_argument("--num-workers", type=int, default=0)
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


def build_dataloaders(data_root: str, batch_size: int, val_split: float, num_workers: int):
	train_split_dir = Path(data_root) / "train"
	dataset = LocalFashionMNIST(train_split_dir, split="train")
	val_size = int(len(dataset) * val_split)
	train_size = len(dataset) - val_size
	train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
	pin_memory = torch.cuda.is_available()
	persistent_workers = num_workers > 0

	train_loader = DataLoader(
		train_dataset,
		batch_size=batch_size,
		shuffle=True,
		num_workers=num_workers,
		pin_memory=pin_memory,
		persistent_workers=persistent_workers,
	)
	val_loader = DataLoader(
		val_dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers,
		pin_memory=pin_memory,
		persistent_workers=persistent_workers,
	)
	return train_loader, val_loader


def build_model(device: torch.device) -> VisionTransformer:
	model = VisionTransformer(
		img_size=28,
		patch_size=7,
		in_channels=1,
		num_classes=10,
		embed_dim=64,
		depth=2,
		num_heads=4,
		mlp_ratio=4,
		dropout=0.1,
	)
	return model.to(device)


def load_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device):
	checkpoint = torch.load(path, map_location=device)
	model.load_state_dict(checkpoint["model_state_dict"])
	optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
	start_epoch = int(checkpoint.get("epoch", 0))
	best_val_acc = float(checkpoint.get("best_val_acc", 0.0))
	return start_epoch, best_val_acc


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
	if train:
		model.train()
	else:
		model.eval()

	total_loss = 0.0
	total_correct = 0
	total_samples = 0

	context = torch.enable_grad() if train else torch.no_grad()
	with context:
		iterator = tqdm(loader, desc="train" if train else "val", leave=False)
		for images, labels in iterator:
			images = images.to(device, non_blocking=True)
			labels = labels.to(device, non_blocking=True)

			if train:
				optimizer.zero_grad(set_to_none=True)

			logits = model(images)
			loss = criterion(logits, labels)

			if train:
				loss.backward()
				optimizer.step()

			batch_size = images.size(0)
			total_loss += loss.item() * batch_size
			total_correct += (logits.argmax(dim=1) == labels).sum().item()
			total_samples += batch_size

			iterator.set_postfix(loss=loss.item(), acc=total_correct / max(total_samples, 1))

	avg_loss = total_loss / max(total_samples, 1)
	avg_acc = total_correct / max(total_samples, 1)
	return avg_loss, avg_acc


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, best_val_acc: float) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	torch.save(
		{
			"epoch": epoch,
			"model_state_dict": model.state_dict(),
			"optimizer_state_dict": optimizer.state_dict(),
			"best_val_acc": best_val_acc,
		},
		path,
	)


def make_timestamped_checkpoint_path(checkpoint_dir: Path, checkpoint_prefix: str) -> Path:
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	return checkpoint_dir / f"{checkpoint_prefix}_{timestamp}.pth"


def main() -> None:
	args = parse_args()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	if device.type == "cuda":
		torch.backends.cudnn.benchmark = True
	train_loader, val_loader = build_dataloaders(args.data_root, args.batch_size, args.val_split, args.num_workers)
	model = build_model(device)
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
	checkpoint_dir = Path(args.checkpoint_dir)
	checkpoint_path = make_timestamped_checkpoint_path(checkpoint_dir, args.checkpoint_prefix)

	start_epoch = 0
	best_val_acc = 0.0
	best_epoch = 0
	if args.resume_checkpoint is not None:
		resume_path = Path(args.resume_checkpoint)
		if not resume_path.exists():
			raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
		start_epoch, best_val_acc = load_checkpoint(resume_path, model, optimizer, device)
		best_epoch = start_epoch
		print(f"Resumed from checkpoint: {resume_path}")
		print(f"Resumed at epoch: {start_epoch} | Best val acc so far: {best_val_acc:.4f}")

	print(f"Device: {device}")
	print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
	print(f"Checkpoint file: {checkpoint_path}")

	for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
		train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
		val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

		print(
			f"Epoch {epoch:02d} | "
			f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
			f"val loss {val_loss:.4f} acc {val_acc:.4f}"
		)

		if val_acc >= best_val_acc:
			best_val_acc = val_acc
			best_epoch = epoch
			save_checkpoint(checkpoint_path, model, optimizer, epoch, best_val_acc)

	print(f"Best checkpoint saved to: {checkpoint_path}")
	print(f"Best epoch: {best_epoch} | Best val acc: {best_val_acc:.4f}")


if __name__ == "__main__":
	main()
