"""训练一个小型 Vision Transformer 来分类 Fashion-MNIST。

这个副本保留原始逻辑，同时尽量按行补充中文注释，方便逐行理解每个模块的作用。
"""

from __future__ import annotations

# 命令行参数解析。
import argparse
# 读取 gzip 压缩的 IDX 数据文件。
import gzip
# 生成时间戳，方便保存带时间的 checkpoint 文件名。
from datetime import datetime
# 按固定字节格式解析 IDX 文件头。
import struct
# 处理文件路径。
from pathlib import Path

# 数值计算和数据格式转换。
import numpy as np
# PyTorch 主框架。
import torch
# 神经网络模块。
from torch import nn
# 数据集、数据加载器和数据集拆分工具。
from torch.utils.data import DataLoader, Dataset, random_split
# 进度条显示。
from tqdm import tqdm

# 导入 ViT 模型定义。
from vit_model import VisionTransformer


def parse_args() -> argparse.Namespace:
	# 创建参数解析器。
	parser = argparse.ArgumentParser(description="Train a small ViT on Fashion-MNIST")
	# 数据根目录，默认指向本地 FashionMNIST 数据。
	parser.add_argument("--data-root", type=str, default="data/FashionMNIST")
	# checkpoint 保存目录。
	parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
	# checkpoint 文件名前缀。
	parser.add_argument("--checkpoint-prefix", type=str, default="vit_fashionmnist")
	# 从指定 checkpoint 继续训练。
	parser.add_argument("--resume-checkpoint", type=str, default=None)
	# 每个 batch 的样本数。
	parser.add_argument("--batch-size", type=int, default=64)
	# 训练轮数。
	parser.add_argument("--epochs", type=int, default=5)
	# 学习率。
	parser.add_argument("--learning-rate", type=float, default=3e-4)
	# 权重衰减。
	parser.add_argument("--weight-decay", type=float, default=0.05)
	# 验证集比例。
	parser.add_argument("--val-split", type=float, default=0.1)
	# DataLoader 使用的 worker 数。
	parser.add_argument("--num-workers", type=int, default=0)
	# 返回解析后的参数对象。
	return parser.parse_args()


def _open_idx_file(path: Path):
	# 根据后缀决定是否按 gzip 打开 IDX 文件。
	return gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb")


def read_idx_images(path: Path) -> torch.Tensor:
	# 读取 IDX 图像文件头和数据体。
	with _open_idx_file(path) as file_obj:
		# 魔数、样本数、高、宽。
		magic, num_images, rows, cols = struct.unpack(">IIII", file_obj.read(16))
		# 校验图像文件魔数，避免读错文件。
		if magic != 2051:
			raise ValueError(f"Unexpected image file magic number: {magic}")
		# 剩余部分就是像素数据。
		buffer = file_obj.read()
	# 将原始字节转换成 NumPy 数组并恢复为 [N, H, W]。
	images = np.frombuffer(buffer, dtype=np.uint8).reshape(num_images, rows, cols).copy()
	# 转成 float Tensor，并归一化到 [0, 1]。
	return torch.from_numpy(images).float().div(255.0)


def read_idx_labels(path: Path) -> torch.Tensor:
	# 读取 IDX 标签文件头和数据体。
	with _open_idx_file(path) as file_obj:
		# 魔数和标签数量。
		magic, num_labels = struct.unpack(">II", file_obj.read(8))
		# 校验标签文件魔数。
		if magic != 2049:
			raise ValueError(f"Unexpected label file magic number: {magic}")
		# 剩余部分就是标签字节。
		buffer = file_obj.read()
	# 转成 NumPy 数组并复制，避免引用底层只读缓冲区。
	labels = np.frombuffer(buffer, dtype=np.uint8).copy()
	# 校验标签数量是否和文件头一致。
	if len(labels) != num_labels:
		raise ValueError("Label file size does not match header")
	# 转成长整型 Tensor，供分类损失函数使用。
	return torch.from_numpy(labels).long()


class LocalFashionMNIST(Dataset):
	def __init__(self, split_dir: Path, split: str) -> None:
		# 统一把输入路径转成 Path 对象。
		split_dir = Path(split_dir)
		# 根据 split 选择训练集或测试集文件名。
		if split == "train":
			image_path = split_dir / "train-images-idx3-ubyte.gz"
			label_path = split_dir / "train-labels-idx1-ubyte.gz"
		elif split == "test":
			image_path = split_dir / "t10k-images-idx3-ubyte.gz"
			label_path = split_dir / "t10k-labels-idx1-ubyte.gz"
		else:
			# 限制 split 只能是 train 或 test。
			raise ValueError("split must be 'train' or 'test'")

		# 检查数据文件是否真的存在。
		if not image_path.exists() or not label_path.exists():
			raise FileNotFoundError(
				f"Missing Fashion-MNIST files in {split_dir}. Expected {image_path.name} and {label_path.name}."
			)

		# 读取图像和标签。
		self.images = read_idx_images(image_path)
		self.labels = read_idx_labels(label_path)

		# 再次确认图像数量和标签数量一致。
		if len(self.images) != len(self.labels):
			raise ValueError("Image count does not match label count")

	def __len__(self) -> int:
		# 数据集长度等于标签数量。
		return len(self.labels)

	def __getitem__(self, index: int):
		# 取出单张图像，并补上通道维度 [1, H, W]。
		image = self.images[index].unsqueeze(0)
		# 取出对应标签。
		label = self.labels[index]
		# 返回一个训练样本。
		return image, label


def build_dataloaders(data_root: str, batch_size: int, val_split: float, num_workers: int):
	# 训练集文件夹路径。
	train_split_dir = Path(data_root) / "train"
	# 加载本地训练集。
	dataset = LocalFashionMNIST(train_split_dir, split="train")
	# 按比例划分验证集大小。
	val_size = int(len(dataset) * val_split)
	# 剩余部分作为训练集。
	train_size = len(dataset) - val_size
	# 随机切分训练集和验证集。
	train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
	# 有 CUDA 时启用 pinned memory，提高数据拷贝效率。
	pin_memory = torch.cuda.is_available()
	# 只有 worker 大于 0 时才启用 persistent workers。
	persistent_workers = num_workers > 0

	# 构建训练 DataLoader。
	train_loader = DataLoader(
		train_dataset,
		batch_size=batch_size,
		shuffle=True,
		num_workers=num_workers,
		pin_memory=pin_memory,
		persistent_workers=persistent_workers,
	)
	# 构建验证 DataLoader。
	val_loader = DataLoader(
		val_dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers,
		pin_memory=pin_memory,
		persistent_workers=persistent_workers,
	)
	# 返回训练集和验证集加载器。
	return train_loader, val_loader


def build_model(device: torch.device) -> VisionTransformer:
	# 按脚本里固定的小型 ViT 配置构建模型。
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
	# 把模型移动到指定设备。
	return model.to(device)


def load_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device):
	# 将 checkpoint 加载到当前设备。
	checkpoint = torch.load(path, map_location=device)
	# 恢复模型参数。
	model.load_state_dict(checkpoint["model_state_dict"])
	# 恢复优化器状态。
	optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
	# 读取已训练到的 epoch。
	start_epoch = int(checkpoint.get("epoch", 0))
	# 读取历史最佳验证精度。
	best_val_acc = float(checkpoint.get("best_val_acc", 0.0))
	# 返回恢复状态。
	return start_epoch, best_val_acc


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
	# 训练阶段启用 train 模式，验证阶段启用 eval 模式。
	if train:
		model.train()
	else:
		model.eval()

	# 统计整轮的损失和正确数。
	total_loss = 0.0
	total_correct = 0
	total_samples = 0

	# 训练阶段需要梯度，验证阶段不需要。
	context = torch.enable_grad() if train else torch.no_grad()
	with context:
		# 用 tqdm 显示批次进度。
		iterator = tqdm(loader, desc="train" if train else "val", leave=False)
		for images, labels in iterator:
			# 把输入数据移动到目标设备。
			images = images.to(device, non_blocking=True)
			labels = labels.to(device, non_blocking=True)

			# 训练时先清空梯度。
			if train:
				optimizer.zero_grad(set_to_none=True)

			# 前向计算得到 logits。
			logits = model(images)
			# 计算分类损失。
			loss = criterion(logits, labels)

			# 只有训练阶段才反向传播并更新参数。
			if train:
				loss.backward()
				optimizer.step()

			# 当前 batch 的样本数。
			batch_size = images.size(0)
			# 按样本数累计损失。
			total_loss += loss.item() * batch_size
			# 累计分类正确的样本数。
			total_correct += (logits.argmax(dim=1) == labels).sum().item()
			# 累计样本总数。
			total_samples += batch_size

			# 实时显示当前 loss 和 acc。
			iterator.set_postfix(loss=loss.item(), acc=total_correct / max(total_samples, 1))

	# 计算整轮平均损失。
	avg_loss = total_loss / max(total_samples, 1)
	# 计算整轮平均准确率。
	avg_acc = total_correct / max(total_samples, 1)
	# 返回整轮指标。
	return avg_loss, avg_acc


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, best_val_acc: float) -> None:
	# 确保保存目录存在。
	path.parent.mkdir(parents=True, exist_ok=True)
	# 把训练状态保存到磁盘。
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
	# 生成当前时间戳字符串。
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	# 拼出最终 checkpoint 路径。
	return checkpoint_dir / f"{checkpoint_prefix}_{timestamp}.pth"


def main() -> None:
	# 解析命令行参数。
	args = parse_args()
	# 优先使用 GPU，否则回退到 CPU。
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	# CUDA 下开启 cudnn benchmark，通常能提升固定输入尺寸的速度。
	if device.type == "cuda":
		torch.backends.cudnn.benchmark = True
	# 构建训练和验证数据加载器。
	train_loader, val_loader = build_dataloaders(args.data_root, args.batch_size, args.val_split, args.num_workers)
	# 构建模型。
	model = build_model(device)
	# 分类损失函数。
	criterion = nn.CrossEntropyLoss()
	# 使用 AdamW 优化器。
	optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
	# checkpoint 保存目录。
	checkpoint_dir = Path(args.checkpoint_dir)
	# 生成一个带时间戳的 checkpoint 文件名。
	checkpoint_path = make_timestamped_checkpoint_path(checkpoint_dir, args.checkpoint_prefix)

	# 默认从第 0 轮开始。
	start_epoch = 0
	# 默认最佳验证精度为 0。
	best_val_acc = 0.0
	# 记录最佳 epoch。
	best_epoch = 0
	# 如果指定了断点续训文件，就先恢复状态。
	if args.resume_checkpoint is not None:
		# 断点文件路径。
		resume_path = Path(args.resume_checkpoint)
		# 文件不存在时直接报错。
		if not resume_path.exists():
			raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
		# 恢复模型和优化器。
		start_epoch, best_val_acc = load_checkpoint(resume_path, model, optimizer, device)
		# 恢复后的最佳 epoch 先按当前 epoch 记录。
		best_epoch = start_epoch
		# 打印恢复信息。
		print(f"Resumed from checkpoint: {resume_path}")
		print(f"Resumed at epoch: {start_epoch} | Best val acc so far: {best_val_acc:.4f}")

	# 打印运行设备。
	print(f"Device: {device}")
	# 打印训练和验证批次数量。
	print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
	# 打印 checkpoint 文件路径。
	print(f"Checkpoint file: {checkpoint_path}")

	# 逐轮训练。
	for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
		# 训练一轮并返回训练集指标。
		train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
		# 在验证集上评估一轮。
		val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

		# 打印当前 epoch 的训练和验证结果。
		print(
			f"Epoch {epoch:02d} | "
			f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
			f"val loss {val_loss:.4f} acc {val_acc:.4f}"
		)

		# 如果当前验证精度更高，就保存为最佳 checkpoint。
		if val_acc >= best_val_acc:
			best_val_acc = val_acc
			best_epoch = epoch
			save_checkpoint(checkpoint_path, model, optimizer, epoch, best_val_acc)

	# 训练结束后打印最终结果。
	print(f"Best checkpoint saved to: {checkpoint_path}")
	print(f"Best epoch: {best_epoch} | Best val acc: {best_val_acc:.4f}")


if __name__ == "__main__":
	# 直接运行脚本时进入主流程。
	main()
