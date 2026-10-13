"""Random batch evaluation for the toy ViT on Fashion-MNIST.

The script samples a configurable number of test images, prints the sample index,
true label and prediction for each one, and then reports the final accuracy.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch

from test import LocalFashionMNIST, find_latest_checkpoint, load_model


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Randomly evaluate a few Fashion-MNIST test images")
	parser.add_argument("--data-root", type=str, default="data/FashionMNIST")
	parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
	parser.add_argument("--checkpoint-prefix", type=str, default="vit_fashionmnist")
	parser.add_argument("--checkpoint", type=str, default=None)
	parser.add_argument("--dataset", "--dataset-split", dest="dataset", type=str, choices=["train", "test"], default="test")
	parser.add_argument("--num-samples", type=int, default=10)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	checkpoint_path = Path(args.checkpoint) if args.checkpoint is not None else find_latest_checkpoint(
		Path(args.checkpoint_dir), args.checkpoint_prefix
	)
	model = load_model(checkpoint_path, device)
	split_dir = Path(args.data_root) / args.dataset
	dataset = LocalFashionMNIST(split_dir, split=args.dataset)

	sample_count = min(args.num_samples, len(dataset))
	if sample_count <= 0:
		raise ValueError("num-samples must be greater than 0")

	indices = random.sample(range(len(dataset)), sample_count)
	correct_count = 0

	print(f"sample count    : {sample_count}")
	print(f"dataset size    : {len(dataset)}")
	print(f"dataset split   : {args.dataset}")
	print(f"checkpoint      : {checkpoint_path}")
	print("-" * 60)

	with torch.no_grad():
		for index in indices:
			image, label = dataset[index]
			image = image.unsqueeze(0).to(device)
			label_int = int(label.item())

			logits = model(image)
			prediction = int(logits.argmax(dim=1).item())
			is_correct = prediction == label_int
			if is_correct:
				correct_count += 1

			print(
				f"index={index:5d} | true label={label_int} | prediction={prediction} | correct={is_correct}"
			)

	accuracy = correct_count / sample_count
	print("-" * 60)
	print(f"correct count   : {correct_count}/{sample_count}")
	print(f"accuracy        : {accuracy:.2%}")


if __name__ == "__main__":
	main()