# ViT 训练、Checkpoint 保存与测试加载流程

本文档说明一张 Fashion-MNIST 图片如何在训练中更新模型参数，并说明 `.pth` checkpoint 文件中保存了什么、测试时如何读回这些参数。

## 1. Checkpoint 是什么

训练脚本保存的 checkpoint 是一个 PyTorch `.pth` 文件，它不是 encoder 本身，而是一次训练状态快照。

保存位置由训练脚本生成，通常类似：

```text
checkpoints/vit_fashionmnist_20260628_032209.pth
```

checkpoint 的保存实现位置：

- `train_vit.py`（line 187）：定义 `save_checkpoint(...)`。
- `train_vit.py`（line 189）：调用 `torch.save(...)` 写入 `.pth` 文件。
- `train_vit.py`（line 192）：保存 `model_state_dict`，也就是模型所有可学习参数。
- `train_vit.py`（line 193）：保存 `optimizer_state_dict`，也就是优化器状态。
- `train_vit.py`（line 246）：当验证集准确率达到当前最好水平时调用 `save_checkpoint(...)`。

保存内容如下：

```python
{
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "best_val_acc": best_val_acc,
}
```

其中最重要的是：

- `model_state_dict`：ViT 模型参数，包括 patch embedding、class token、positional embedding、Transformer blocks、LayerNorm、classification head。
- `optimizer_state_dict`：AdamW 优化器状态，包括历史动量等信息，用于断点续训。
- `epoch`：保存时训练到了第几轮。
- `best_val_acc`：保存时最好的验证集准确率。

## 2. 模型中哪些参数会被保存

模型定义在 `vit_model.py`。

主要可学习参数来源：

- `vit_model.py`（line 29）：`PatchEmbedding` 中的 `Conv2d`，保存 `patch_embed.proj.weight` 和 `patch_embed.proj.bias`。
- `vit_model.py`（line 54）：self-attention 的 `qkv` 线性层，保存 Q/K/V 投影参数。
- `vit_model.py`（line 56）：self-attention 输出投影层，保存 attention 输出融合参数。
- `vit_model.py`（line 102）：Transformer block 中的第一个 `LayerNorm` 参数。
- `vit_model.py`（line 103）：Transformer block 中的 `MultiHeadSelfAttention` 参数。
- `vit_model.py`（line 104）：Transformer block 中的第二个 `LayerNorm` 参数。
- `vit_model.py`（line 105）：Transformer block 中的 MLP 参数。
- `vit_model.py`（line 136）：`cls_token`，分类用的可学习 token。
- `vit_model.py`（line 137）：`pos_embed`，可学习位置编码。
- `vit_model.py`（line 140）：多个 Transformer blocks 的参数集合。
- `vit_model.py`（line 151）：最终 `LayerNorm` 参数。
- `vit_model.py`（line 152）：classification head 参数，把特征映射到 10 个类别。

这些参数都会包含在 `model.state_dict()` 中，并在 `train_vit.py`（line 192）写入 `.pth`。

## 3. 训练时参数如何更新

### 3.1 创建模型

模型在训练脚本中创建：

- `train_vit.py`（line 124）：定义 `build_model(device)`。
- `train_vit.py`（line 125）：实例化 `VisionTransformer(...)`。

核心模型参数配置：

```text
img_size=28
patch_size=7
in_channels=1
num_classes=10
embed_dim=64
depth=2
num_heads=4
mlp_ratio=4
```

这意味着：

- 输入图片 shape 是 `[B, 1, 28, 28]`。
- patch 大小是 `7x7`，所以每张图被切成 `4x4=16` 个 patch。
- 每个 patch token 是 `64` 维。
- Transformer block 堆叠 `2` 层。
- 每层 self-attention 有 `4` 个 head，每个 head 维度是 `64 / 4 = 16`。
- 输出类别数是 `10`。

### 3.2 创建 loss 和 optimizer

训练中用交叉熵损失和 AdamW 优化器：

- `train_vit.py`（line 212）：`criterion = nn.CrossEntropyLoss()`。
- `train_vit.py`（line 213）：`optimizer = torch.optim.AdamW(...)`。

`CrossEntropyLoss` 接收：

```text
logits shape: [B, 10]
labels shape: [B]
loss shape: 标量
```

`AdamW` 负责根据梯度更新模型参数。

### 3.3 进入训练模式

每个 epoch 调用：

- `train_vit.py`（line 148）：定义 `run_epoch(...)`。
- `train_vit.py`（line 150）：训练时执行 `model.train()`。
- `train_vit.py`（line 152）：验证时执行 `model.eval()`。

`model.train()` 会启用训练行为，例如 dropout。

`model.eval()` 会关闭 dropout 的随机行为，用于稳定验证或测试。

### 3.4 DataLoader 取出 batch

训练循环中：

- `train_vit.py`（line 161）：`for images, labels in iterator:`。
- `train_vit.py`（line 162）：把 `images` 移动到 CPU 或 GPU。
- `train_vit.py`（line 163）：把 `labels` 移动到 CPU 或 GPU。

此时张量含义：

```text
images shape: [B, 1, 28, 28]
labels shape: [B]
```

其中：

- `B` 是 batch size，默认是 64。
- `1` 是灰度图通道数。
- `28, 28` 是图片高宽。
- `labels` 中每个值是 0 到 9 的类别编号。

### 3.5 清空上一轮梯度

参数更新前先清空旧梯度：

- `train_vit.py`（line 166）：`optimizer.zero_grad(set_to_none=True)`。

原因是 PyTorch 默认会累积梯度。如果不清空，当前 batch 的梯度会和上一个 batch 的梯度叠加。

### 3.6 前向传播得到 logits

训练脚本调用模型：

- `train_vit.py`（line 168）：`logits = model(images)`。

这会进入：

- `vit_model.py`（line 173）：`VisionTransformer.forward(...)`。

完整 shape 流程如下：

```text
输入 images: [B, 1, 28, 28]
patch embedding 后: [B, 16, 64]
加入 cls token 后: [B, 17, 64]
加入 positional embedding 后: [B, 17, 64]
经过 2 个 Transformer block 后: [B, 17, 64]
取 class token 后: [B, 64]
classification head 后: [B, 10]
```

关键实现位置：

- `vit_model.py`（line 174）：调用 `self.patch_embed(x)`。
- `vit_model.py`（line 177）：扩展 `cls_token` 到当前 batch。
- `vit_model.py`（line 179）：加入 `pos_embed`。
- `vit_model.py`（line 182）：依次经过 Transformer blocks。
- `vit_model.py`（line 185）：最终 LayerNorm。
- `vit_model.py`（line 187）：classification head 输出 logits。

### 3.7 计算 loss

训练脚本计算损失：

- `train_vit.py`（line 169）：`loss = criterion(logits, labels)`。

含义：

```text
logits: [B, 10]
labels: [B]
loss: 一个标量
```

`CrossEntropyLoss` 会比较模型输出的 10 类分数和真实类别，得到当前 batch 的分类损失。

### 3.8 反向传播计算梯度

训练脚本执行：

- `train_vit.py`（line 172）：`loss.backward()`。

这一步不会直接修改参数，而是根据 loss 沿计算图反向计算每个可学习参数的梯度。

会得到梯度的参数包括：

- `patch_embed.proj.weight`
- `patch_embed.proj.bias`
- `cls_token`
- `pos_embed`
- `blocks.*.attn.qkv.weight`
- `blocks.*.attn.qkv.bias`
- `blocks.*.attn.proj.weight`
- `blocks.*.attn.proj.bias`
- `blocks.*.mlp.*`
- `blocks.*.norm*`
- `norm.weight`
- `norm.bias`
- `head.weight`
- `head.bias`

### 3.9 optimizer step 更新参数

真正修改模型参数的位置：

- `train_vit.py`（line 173）：`optimizer.step()`。

这一步使用 AdamW 根据当前梯度更新模型参数。

简化理解：

```text
旧参数 + 当前梯度 + AdamW 优化器状态 -> 新参数
```

更新后的新参数仍然存在内存中的 `model` 里。

### 3.10 保存更新后的参数到 `.pth`

每个 epoch 训练和验证后，如果验证准确率达到当前最好水平，就保存 checkpoint：

- `train_vit.py`（line 246）：调用 `save_checkpoint(...)`。
- `train_vit.py`（line 192）：把更新后的 `model.state_dict()` 写入 `model_state_dict`。
- `train_vit.py`（line 193）：把更新后的 `optimizer.state_dict()` 写入 `optimizer_state_dict`。

所以 `.pth` 中保存的是 optimizer step 之后的参数，而不是初始随机参数。

## 4. 测试时参数如何从 `.pth` 加载

测试脚本不会更新参数，只会加载参数并做推理。

### 4.1 找到 checkpoint

测试脚本先决定使用哪个 `.pth` 文件：

- `test.py`（line 59）：定义 `find_latest_checkpoint(...)`。
- `test.py`（line 188）：如果没有手动指定 checkpoint，就自动查找最新 checkpoint。

### 4.2 创建同结构模型

测试脚本中：

- `test.py`（line 107）：定义 `load_model(...)`。
- `test.py`（line 108）：创建新的 `VisionTransformer()`。

注意：这里必须创建和训练时结构一致的模型。否则 `.pth` 中的参数名和 shape 可能对不上。

### 4.3 从 `.pth` 读取参数

测试脚本加载 checkpoint：

- `test.py`（line 112）：`checkpoint = torch.load(checkpoint_path, map_location=device)`。
- `test.py`（line 113）：从 checkpoint 中取 `model_state_dict`。
- `test.py`（line 114）：`model.load_state_dict(state_dict)` 把训练好的参数写回模型。

这一步之后，测试模型就拥有训练好的权重。

### 4.4 切换到 eval 模式

测试脚本执行：

- `test.py`（line 119）：`model.eval()`。

这会关闭 dropout 等训练时随机行为，让同一张图的推理结果更稳定。

### 4.5 测试时不计算梯度

测试脚本执行：

- `test.py`（line 198）：`with torch.no_grad():`。

这表示测试阶段不记录梯度，不做反向传播，也不会更新参数。

### 4.6 前向推理得到预测

测试脚本拆解前向流程：

- `test.py`（line 158）：定义 `describe_forward(...)`。
- `test.py`（line 199）：调用 `describe_forward(model, image)` 得到 logits。
- `test.py`（line 200）：`prediction = logits.argmax(dim=1).item()` 得到预测类别。

测试阶段 shape 对单张图片通常是：

```text
输入 image: [1, 1, 28, 28]
patch embedding 后: [1, 16, 64]
加入 cls token 后: [1, 17, 64]
Transformer blocks 后: [1, 17, 64]
取 class token 后: [1, 64]
classification head 后 logits: [1, 10]
prediction: 一个整数类别
```

测试时没有：

```python
loss.backward()
optimizer.step()
```

所以测试不会改变 `.pth` 文件，也不会改变模型参数。

## 5. 训练和测试的关键区别

| 阶段 | 是否加载 checkpoint | 是否计算 loss | 是否 backward | 是否 optimizer step | 是否保存 checkpoint |
| --- | --- | --- | --- | --- | --- |
| 训练 | 可选，断点续训时加载 | 是 | 是 | 是 | 是 |
| 验证 | 使用当前内存中的模型参数 | 是 | 否 | 否 | 准确率最好时保存 |
| 测试 | 是 | 否 | 否 | 否 | 否 |

关键实现位置：

- 训练前向：`train_vit.py`（line 168）。
- 训练 loss：`train_vit.py`（line 169）。
- 训练 backward：`train_vit.py`（line 172）。
- 训练参数更新：`train_vit.py`（line 173）。
- checkpoint 保存：`train_vit.py`（line 189）。
- 测试 checkpoint 加载：`test.py`（line 112）到 `test.py`（line 114）。
- 测试关闭梯度：`test.py`（line 198）。

## 6. 一句话总结

训练时，图片 batch `[B, 1, 28, 28]` 经过 ViT 得到 logits `[B, 10]`，`CrossEntropyLoss` 根据 logits 和 labels 计算损失，`loss.backward()` 计算梯度，`optimizer.step()` 更新模型参数，表现最好时用 `torch.save(...)` 把更新后的参数写入 `.pth`。

测试时，脚本创建同结构 ViT，用 `torch.load(...)` 读取 `.pth`，再用 `model.load_state_dict(...)` 把训练好的参数装回模型，然后只做前向推理，不做参数更新。
