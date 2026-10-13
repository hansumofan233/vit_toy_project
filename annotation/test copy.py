"""逐步检查单张 Fashion-MNIST 图片经过 ViT 的调试脚本。

这个文件是 test.py 的中文逐行解释副本，
主要用于配合 VS Code 断点观察每一步张量形状和变量变化。
"""

from __future__ import annotations  # 延迟解析类型注解，减少运行时对类型名定义顺序的依赖。

import argparse  # 用于解析命令行参数，例如 --dataset、--index。
import gzip  # 用于读取 .gz 压缩格式的 Fashion-MNIST 原始数据文件。
import random  # 用于在没有指定 index 时随机抽取一张样本。
import struct  # 用于按二进制格式解析 IDX 文件头。
from pathlib import Path  # 用 Path 对象处理文件路径，比字符串路径更清晰。

import numpy as np  # 用 NumPy 把二进制图片和标签缓冲区转成数组。
import matplotlib.pyplot as plt  # 用 matplotlib 显示单张 Fashion-MNIST 图片。
import torch  # 导入 PyTorch，用于张量计算和模型推理。
from torch.utils.data import Dataset  # 导入 Dataset 基类，用来封装本地数据集。

from vit_model import VisionTransformer  # 导入当前项目里的 Vision Transformer 模型。


def parse_args() -> argparse.Namespace:  # 定义命令行参数解析函数，返回 argparse 的参数对象。
    parser = argparse.ArgumentParser(description="Debug one ViT forward pass")  # 创建参数解析器并设置说明文字。
    parser.add_argument("--data-root", type=str, default="data/FashionMNIST")  # 设置 Fashion-MNIST 数据根目录。
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")  # 设置模型权重所在目录。
    parser.add_argument("--checkpoint-prefix", type=str, default="vit_fashionmnist")  # 设置自动查找权重时的文件名前缀。
    parser.add_argument("--checkpoint", type=str, default=None)  # 允许手动指定某个 checkpoint 文件路径。
    parser.add_argument(  # 添加数据集划分参数。
        "--dataset",  # 参数名之一，例如 --dataset test。
        "--dataset-split",  # 参数别名，例如 --dataset-split train。
        dest="dataset",  # 解析后统一保存到 args.dataset。
        type=str,  # 参数值类型是字符串。
        choices=["train", "test"],  # 只允许 train 或 test 两种取值。
        default="test",  # 默认使用测试集。
    )
    parser.add_argument("--index", type=int, default=None)  # 指定样本下标；不指定时随机抽样。
    return parser.parse_args()  # 解析终端传入的参数并返回。


def _open_idx_file(path: Path):  # 定义一个小工具函数，用来打开 IDX 文件。
    return gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb")  # gz 文件用 gzip 打开，否则普通二进制打开。


def read_idx_images(path: Path) -> torch.Tensor:  # 读取 IDX 图片文件，并返回 PyTorch 张量。
    with _open_idx_file(path) as file_obj:  # 打开图片文件，with 会在读取完成后自动关闭文件。
        magic, num_images, rows, cols = struct.unpack(">IIII", file_obj.read(16))  # 读取 16 字节文件头并解析元信息。
        if magic != 2051:  # 图片文件的 magic number 应该是 2051。
            raise ValueError(f"Unexpected image file magic number: {magic}")  # 如果文件类型不对，直接抛出错误。
        buffer = file_obj.read()  # 读取剩余的全部图片像素数据。
    images = np.frombuffer(buffer, dtype=np.uint8).reshape(num_images, rows, cols).copy()  # 转成 [N, H, W] 的 uint8 数组。
    return torch.from_numpy(images).float().div(255.0)  # 转成 float 张量，并把像素归一化到 0 到 1。


def read_idx_labels(path: Path) -> torch.Tensor:  # 读取 IDX 标签文件，并返回 PyTorch 张量。
    with _open_idx_file(path) as file_obj:  # 打开标签文件。
        magic, num_labels = struct.unpack(">II", file_obj.read(8))  # 读取 8 字节文件头，解析 magic number 和标签数量。
        if magic != 2049:  # 标签文件的 magic number 应该是 2049。
            raise ValueError(f"Unexpected label file magic number: {magic}")  # 如果不是标签文件，抛出错误。
        buffer = file_obj.read()  # 读取剩余的全部标签数据。
    labels = np.frombuffer(buffer, dtype=np.uint8).copy()  # 把标签缓冲区转成 uint8 数组。
    if len(labels) != num_labels:  # 检查实际读取到的标签数是否和文件头一致。
        raise ValueError("Label file size does not match header")  # 数量不一致说明文件可能损坏。
    return torch.from_numpy(labels).long()  # 转成 long 类型张量，适合作为分类标签。


def find_latest_checkpoint(checkpoint_dir: Path, checkpoint_prefix: str) -> Path:  # 自动查找最新的模型权重文件。
    checkpoint_dir = Path(checkpoint_dir)  # 确保 checkpoint_dir 是 Path 对象。
    pattern = f"{checkpoint_prefix}_*.pth"  # 拼出带时间戳权重文件的匹配模式。
    candidates = sorted(checkpoint_dir.glob(pattern), key=lambda path: path.stat().st_mtime)  # 按修改时间排序候选权重。
    if candidates:  # 如果找到了带时间戳的权重文件。
        return candidates[-1]  # 返回修改时间最新的一个。

    fallback = checkpoint_dir / f"{checkpoint_prefix}.pth"  # 如果没有时间戳版本，就尝试固定文件名。
    if fallback.exists():  # 检查固定文件名权重是否存在。
        return fallback  # 存在就返回这个 fallback 文件。

    raise FileNotFoundError(  # 如果两种方式都找不到，就抛出文件不存在错误。
        f"No checkpoint found in {checkpoint_dir}. Expected files matching {pattern} or {fallback.name}."
    )


class LocalFashionMNIST(Dataset):  # 定义本地 Fashion-MNIST 数据集类，继承 PyTorch Dataset。
    def __init__(self, split_dir: Path, split: str) -> None:  # 初始化数据集，需要数据目录和 train/test 划分。
        split_dir = Path(split_dir)  # 确保 split_dir 是 Path 对象。
        if split == "train":  # 如果读取训练集。
            image_path = split_dir / "train-images-idx3-ubyte.gz"  # 训练集图片文件路径。
            label_path = split_dir / "train-labels-idx1-ubyte.gz"  # 训练集标签文件路径。
        elif split == "test":  # 如果读取测试集。
            image_path = split_dir / "t10k-images-idx3-ubyte.gz"  # 测试集图片文件路径。
            label_path = split_dir / "t10k-labels-idx1-ubyte.gz"  # 测试集标签文件路径。
        else:  # 如果 split 既不是 train 也不是 test。
            raise ValueError("split must be 'train' or 'test'")  # 抛出参数错误。

        if not image_path.exists() or not label_path.exists():  # 检查图片文件和标签文件是否都存在。
            raise FileNotFoundError(  # 缺任意一个文件就给出明确错误提示。
                f"Missing Fashion-MNIST files in {split_dir}. Expected {image_path.name} and {label_path.name}."
            )

        self.images = read_idx_images(image_path)  # 读取全部图片，形状是 [N, 28, 28]。
        self.labels = read_idx_labels(label_path)  # 读取全部标签，形状是 [N]。

        if len(self.images) != len(self.labels):  # 检查图片数量和标签数量是否一致。
            raise ValueError("Image count does not match label count")  # 不一致说明数据文件不匹配。

    def __len__(self) -> int:  # 定义 len(dataset) 的行为。
        return len(self.labels)  # 数据集长度等于标签数量。

    def __getitem__(self, index: int):  # 定义 dataset[index] 的取样逻辑。
        image = self.images[index].unsqueeze(0)  # 取一张图片并增加通道维，形状从 [28, 28] 变成 [1, 28, 28]。
        label = self.labels[index]  # 取对应标签。
        return image, label  # 返回一对 image 和 label。


def load_model(checkpoint_path: Path, device: torch.device) -> VisionTransformer:  # 加载模型和权重。
    model = VisionTransformer()  # 创建一个默认配置的 ViT 模型。
    model.to(device)  # 把模型移动到 CPU 或 GPU。

    if checkpoint_path.exists():  # 如果 checkpoint 文件存在。
        checkpoint = torch.load(checkpoint_path, map_location=device)  # 加载权重文件，并映射到当前设备。
        state_dict = checkpoint.get("model_state_dict", checkpoint)  # 兼容两种保存格式：完整 checkpoint 或纯 state_dict。
        model.load_state_dict(state_dict)  # 把权重加载进模型。
        print(f"Loaded checkpoint: {checkpoint_path}")  # 打印已加载的权重路径。
    else:  # 如果指定的 checkpoint 不存在。
        print(f"Checkpoint not found, using randomly initialized model: {checkpoint_path}")  # 提示将使用随机初始化模型。

    model.eval()  # 切换到评估模式，关闭 dropout 等训练行为。
    return model  # 返回准备好的模型。


def load_sample(data_root: str, dataset_split: str, index: int | None):  # 从本地数据集中加载一张样本。
    split_dir = Path(data_root) / dataset_split  # 拼出 train 或 test 的数据目录。
    dataset = LocalFashionMNIST(split_dir, split=dataset_split)  # 创建本地 Fashion-MNIST 数据集对象。
    if index is None:  # 如果用户没有指定样本编号。
        index = random.randrange(len(dataset))  # 随机选择一个合法下标。
    image, label = dataset[index]  # 从数据集中取出指定图片和标签。
    return image.unsqueeze(0), label, index  # 给图片增加 batch 维，返回 [1, 1, 28, 28]、标签和下标。


def show_sample_image(image: torch.Tensor, index: int, label: int, prediction: int | None = None) -> None:  # 显示样本图片和预测信息。
    image_2d = image.squeeze(0).squeeze(0).cpu().numpy()  # 去掉 batch 和 channel 维，转成二维 NumPy 数组。
    plt.figure(figsize=(4, 4))  # 创建一个 4x4 英寸的图像窗口。
    plt.imshow(image_2d, cmap="gray")  # 以灰度图方式显示图片。
    if prediction is None:  # 如果还没有预测结果。
        title = f"Fashion-MNIST | index={index} | label={label}"  # 标题只显示样本编号和真实标签。
    else:  # 如果已有预测结果。
        title = f"Fashion-MNIST | index={index} | label={label} | prediction={prediction}"  # 标题同时显示预测标签。
    plt.title(title)  # 设置图像标题。
    plt.text(  # 在图像左上角叠加文字说明。
        0.02,  # 文本 x 坐标，使用坐标轴比例。
        0.96,  # 文本 y 坐标，靠近上方。
        f"label: {label}\nprediction: {prediction if prediction is not None else 'N/A'}",  # 显示真实标签和预测结果。
        transform=plt.gca().transAxes,  # 使用当前坐标轴的相对坐标。
        fontsize=10,  # 设置文字大小。
        color="yellow",  # 设置文字颜色为黄色。
        verticalalignment="top",  # 让文字从顶部开始对齐。
        bbox=dict(facecolor="black", alpha=0.55, pad=4),  # 给文字加半透明黑色背景，方便看清。
    )
    plt.axis("off")  # 关闭坐标轴刻度和边框。
    plt.tight_layout()  # 自动调整布局，减少遮挡。
    plt.show()  # 弹出图像窗口。


def describe_forward(model: VisionTransformer, image: torch.Tensor) -> torch.Tensor:  # 手动拆解 ViT 前向传播，方便逐步调试。
    print(f"input          : {image.shape}")  # 打印输入图片形状，通常是 [1, 1, 28, 28]。
    x = model.patch_embed(image)  # 把图片切成 patch，并投影成 token。
    print(f"patch_embed    : {x.shape}")  # 打印 patch token 形状，通常是 [1, 16, 64]。

    batch_size = x.shape[0]  # 取 batch 大小，这里通常是 1。
    cls_token = model.cls_token.expand(batch_size, -1, -1)  # 把可学习的 cls token 扩展到当前 batch 大小。
    print(f"cls_token      : {cls_token.shape}")  # 打印 cls token 形状，通常是 [1, 1, 64]。
    x = torch.cat((cls_token, x), dim=1)  # 把 cls token 拼到 patch token 序列最前面。
    print(f"after concat   : {x.shape}")  # 打印拼接后的形状，通常是 [1, 17, 64]。
    x = x + model.pos_embed  # 给每个 token 加上对应的位置编码。
    print(f"after pos_embed: {x.shape}")  # 打印加入位置编码后的形状，形状不变。
    x = model.pos_drop(x)  # 对 token 序列做 dropout；eval 模式下通常不会随机丢弃。

    for block_index, block in enumerate(model.blocks, start=1):  # 按顺序遍历每一个 Transformer block。
        x = block(x)  # 让当前 token 序列通过一个 Transformer block。
        print(f"block {block_index:<2}      : {x.shape}")  # 打印当前 block 输出形状。

    x = model.norm(x)  # 对最后一层输出做 LayerNorm。
    print(f"after norm     : {x.shape}")  # 打印归一化后的形状。
    cls_output = x[:, 0]  # 取第 0 个 token，也就是 cls token 的最终表示。
    print(f"cls_output     : {cls_output.shape}")  # 打印 cls token 表示形状，通常是 [1, 64]。
    logits = model.head(cls_output)  # 通过分类头得到每个类别的 logits。
    print(f"logits         : {logits.shape}")  # 打印 logits 形状，通常是 [1, 10]。
    return logits  # 返回分类结果 logits。


def main() -> None:  # 定义脚本主入口。
    args = parse_args()  # 解析命令行参数。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 如果有 CUDA 就用 GPU，否则用 CPU。
    checkpoint_path = Path(args.checkpoint) if args.checkpoint is not None else find_latest_checkpoint(  # 决定要加载哪个权重文件。
        Path(args.checkpoint_dir), args.checkpoint_prefix  # 未手动指定 checkpoint 时，根据目录和前缀自动找最新权重。
    )
    model = load_model(checkpoint_path, device)  # 创建模型并加载权重。
    image, label, sample_index = load_sample(args.data_root, args.dataset, args.index)  # 加载一张样本图片、标签和编号。
    image = image.to(device)  # 把图片移动到和模型相同的设备。

    print(f"sample index    : {sample_index}")  # 打印当前样本编号。
    print(f"dataset split   : {args.dataset}")  # 打印当前使用 train 还是 test。
    print(f"true label      : {label}")  # 打印真实标签。
    with torch.no_grad():  # 关闭梯度计算，节省内存并加快推理。
        logits = describe_forward(model, image)  # 手动执行并打印 ViT 前向传播各阶段形状。
        prediction = logits.argmax(dim=1).item()  # 取 logits 最大的类别作为预测结果。
        print(f"prediction      : {prediction}")  # 打印预测类别。
    show_sample_image(image, sample_index, int(label.item()), prediction)  # 显示图片、真实标签和预测标签。


if __name__ == "__main__":  # 只有直接运行本文件时才执行 main。
    main()  # 调用主函数，启动完整测试流程。
