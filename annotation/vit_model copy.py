"""Vision Transformer toy 模型的中文逐行注释版本。

这个文件保留原始实现的计算逻辑，
重点是在大多数关键代码行旁边补充中文说明，
方便直接对照代码理解每一步在做什么。
"""

from __future__ import annotations  # 让类型注解延迟求值，便于书写和兼容前向引用。

import torch  # 导入 PyTorch 主包，用于张量计算和随机数据生成。
from torch import nn  # 从 torch 中导入神经网络模块，后续可直接写 nn.xxx。


class PatchEmbedding(nn.Module):  # 定义 PatchEmbedding 模块，把图像切成 patch 并编码成 token。
    """把输入图片切成 patch，并映射成 token 向量。"""

    def __init__(
        self,  # 当前模块实例本身。
        img_size: int = 28,  # 输入图像的高和宽，这里默认是 28。
        patch_size: int = 7,  # 每个 patch 的边长，默认把图切成 7x7 的小块。
        in_channels: int = 1,  # 输入通道数，灰度图为 1，彩色图通常为 3。
        embed_dim: int = 64,  # 每个 patch 被映射后的特征维度。
    ) -> None:
        super().__init__()  # 先初始化父类 nn.Module 的内部状态。
        if img_size % patch_size != 0:  # 如果图像尺寸不能整除 patch 尺寸。
            raise ValueError("img_size must be divisible by patch_size")  # 直接报错，避免切 patch 时出现残块。

        self.img_size = img_size  # 保存图像尺寸，便于后续查看配置。
        self.patch_size = patch_size  # 保存 patch 尺寸，便于后续查看配置。
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)  # 计算总 patch 数量。

        self.proj = nn.Conv2d(  # 使用卷积一次性完成 patch 切分和线性投影。
            in_channels=in_channels,  # 卷积输入通道数等于图像通道数。
            out_channels=embed_dim,  # 卷积输出通道数等于嵌入维度。
            kernel_size=patch_size,  # 卷积核大小等于 patch 大小。
            stride=patch_size,  # 步长等于 patch 大小，确保 patch 之间不重叠。
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 定义前向传播，输入是图像张量。
        # 输入张量形状是 [B, C, H, W]，分别表示批大小、通道数、高、宽。
        x = self.proj(x)  # 经过卷积后得到 [B, embed_dim, H/patch, W/patch]。
        x = x.flatten(2)  # 从第 2 维开始展平空间维度，得到 [B, embed_dim, num_patches]。
        x = x.transpose(1, 2)  # 交换维度，整理成 [B, num_patches, embed_dim] 的 token 序列。
        return x  # 返回 patch token 序列。


class MultiHeadSelfAttention(nn.Module):  # 定义多头自注意力模块，用来建模 token 间关系。
    """多头自注意力层，负责在 token 之间建立关联。"""

    def __init__(
        self,  # 当前模块实例本身。
        embed_dim: int,  # 输入 token 的总特征维度。
        num_heads: int,  # 注意力头数。
        dropout: float = 0.1,  # dropout 比例，用于正则化。
    ) -> None:
        super().__init__()  # 初始化父类 nn.Module。
        if embed_dim % num_heads != 0:  # 如果总维度不能均匀分给每个头。
            raise ValueError("embed_dim must be divisible by num_heads")  # 直接报错，避免 reshape 出错。

        self.embed_dim = embed_dim  # 保存总嵌入维度。
        self.num_heads = num_heads  # 保存注意力头数。
        self.head_dim = embed_dim // num_heads  # 计算每个头分到的特征维度。
        self.scale = self.head_dim**-0.5  # 计算缩放因子，防止点积值过大。

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)  # 一次线性映射同时生成 Q、K、V。
        self.attn_drop = nn.Dropout(dropout)  # 对注意力权重做 dropout。
        self.proj = nn.Linear(embed_dim, embed_dim)  # 多头拼接后再做一次线性映射。
        self.proj_drop = nn.Dropout(dropout)  # 对输出投影结果再做 dropout。

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 输入 x 的形状是 [B, tokens, embed_dim]。
        batch_size, num_tokens, embed_dim = x.shape  # 拆出批大小、token 数量和特征维度。

        qkv = self.qkv(x)  # 先把输入映射成拼接在一起的 Q、K、V。
        qkv = qkv.reshape(batch_size, num_tokens, 3, self.num_heads, self.head_dim)  # 拆出 QKV 和多头结构。
        qkv = qkv.permute(2, 0, 3, 1, 4)  # 调整维度顺序，变成 [3, B, heads, tokens, head_dim]。
        q, k, v = qkv[0], qkv[1], qkv[2]  # 分别取出 query、key、value。

        attn = (q @ k.transpose(-2, -1)) * self.scale  # 计算缩放点积注意力分数。
        attn = attn.softmax(dim=-1)  # 在最后一维做 softmax，得到注意力权重。
        attn = self.attn_drop(attn)  # 对注意力权重施加 dropout。

        out = attn @ v  # 用注意力权重对 value 做加权求和。
        out = out.transpose(1, 2).reshape(batch_size, num_tokens, embed_dim)  # 多头结果拼回原始维度。
        out = self.proj(out)  # 经过输出线性层做特征融合。
        out = self.proj_drop(out)  # 对输出结果再做一次 dropout。
        return out  # 返回自注意力模块的输出。


class FeedForward(nn.Module):  # 定义 Transformer 中常见的前馈网络模块。
    """Transformer 中的前馈网络，做非线性特征变换。"""

    def __init__(
        self,  # 当前模块实例本身。
        embed_dim: int,  # 输入和输出的特征维度。
        hidden_dim: int,  # 中间隐藏层维度，通常比 embed_dim 更大。
        dropout: float = 0.1,  # dropout 比例。
    ) -> None:
        super().__init__()  # 初始化父类 nn.Module。
        self.net = nn.Sequential(  # 使用顺序容器把多层按前向顺序串起来。
            nn.Linear(embed_dim, hidden_dim),  # 第一层线性层，先升维。
            nn.GELU(),  # 使用 GELU 激活函数增强非线性表达能力。
            nn.Dropout(dropout),  # 在中间激活后做 dropout。
            nn.Linear(hidden_dim, embed_dim),  # 第二层线性层，把维度降回原始大小。
            nn.Dropout(dropout),  # 输出后再做一次 dropout。
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 定义前向传播。
        return self.net(x)  # 直接按顺序执行前馈网络并返回结果。


class TransformerBlock(nn.Module):  # 定义标准的 Transformer 编码块。
    """一个标准 Transformer Block: LayerNorm + Attention + MLP + 残差连接。"""

    def __init__(
        self,  # 当前模块实例本身。
        embed_dim: int,  # token 特征维度。
        num_heads: int,  # 注意力头数。
        mlp_ratio: int = 4,  # 前馈网络隐藏层维度相对 embed_dim 的放大倍数。
        dropout: float = 0.1,  # dropout 比例。
    ) -> None:
        super().__init__()  # 初始化父类 nn.Module。
        self.norm1 = nn.LayerNorm(embed_dim)  # 注意力前的第一层归一化。
        self.attn = MultiHeadSelfAttention(  # 构建多头自注意力子模块。
            embed_dim=embed_dim,  # 传入总特征维度。
            num_heads=num_heads,  # 传入注意力头数。
            dropout=dropout,  # 传入 dropout 配置。
        )
        self.norm2 = nn.LayerNorm(embed_dim)  # 前馈网络前的第二层归一化。
        self.mlp = FeedForward(  # 构建前馈网络子模块。
            embed_dim=embed_dim,  # 输入输出维度保持为 embed_dim。
            hidden_dim=embed_dim * mlp_ratio,  # 隐藏层维度按倍数放大。
            dropout=dropout,  # 传入 dropout 配置。
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 定义一个 Transformer block 的前向传播。
        x = x + self.attn(self.norm1(x))  # 先归一化，再做注意力，最后与输入做第一次残差相加。
        x = x + self.mlp(self.norm2(x))  # 再归一化，经过 MLP，最后与当前特征做第二次残差相加。
        return x  # 返回经过一个 block 编码后的 token 序列。


class VisionTransformer(nn.Module):  # 定义完整的 Vision Transformer 分类模型。
    """用于小型图像分类任务的简化版 ViT。"""

    def __init__(
        self,  # 当前模型实例本身。
        img_size: int = 28,  # 输入图像边长。
        patch_size: int = 7,  # patch 边长。
        in_channels: int = 1,  # 输入通道数。
        num_classes: int = 10,  # 最终分类类别数。
        embed_dim: int = 64,  # token 嵌入维度。
        depth: int = 2,  # Transformer block 的堆叠层数。
        num_heads: int = 4,  # 每层注意力的头数。
        mlp_ratio: int = 4,  # 前馈网络隐藏层放大倍数。
        dropout: float = 0.1,  # 全模型中使用的 dropout 比例。
    ) -> None:
        super().__init__()  # 初始化父类 nn.Module。

        self.patch_embed = PatchEmbedding(  # 构建 patch 嵌入模块，作为图像到 token 的入口。
            img_size=img_size,  # 传入图像尺寸。
            patch_size=patch_size,  # 传入 patch 尺寸。
            in_channels=in_channels,  # 传入输入通道数。
            embed_dim=embed_dim,  # 传入嵌入维度。
        )
        num_patches = self.patch_embed.num_patches  # 读取总 patch 数，用于后续位置编码长度设置。

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # 定义可学习的分类 token。
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))  # 定义可学习的位置编码。
        self.pos_drop = nn.Dropout(dropout)  # 在加入位置编码后做 dropout。

        self.blocks = nn.ModuleList(  # 用 ModuleList 按顺序保存多个 Transformer block。
            [
                TransformerBlock(  # 为每一层创建一个独立的 Transformer block。
                    embed_dim=embed_dim,  # 每层的特征维度相同。
                    num_heads=num_heads,  # 每层的注意力头数相同。
                    mlp_ratio=mlp_ratio,  # 每层的 MLP 扩张比例相同。
                    dropout=dropout,  # 每层使用同样的 dropout。
                )
                for _ in range(depth)  # 根据 depth 重复创建指定数量的 block。
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)  # 在分类前对最终 token 表示做归一化。
        self.head = nn.Linear(embed_dim, num_classes)  # 分类头，把 cls token 映射到类别数。

        self._init_weights()  # 调用自定义初始化逻辑，初始化所有参数。

    def _init_weights(self) -> None:  # 定义统一的权重初始化函数。
        nn.init.trunc_normal_(self.cls_token, std=0.02)  # 用截断正态分布初始化 cls token。
        nn.init.trunc_normal_(self.pos_embed, std=0.02)  # 用截断正态分布初始化位置编码。

        for module in self.modules():  # 遍历当前模型中的所有子模块。
            if isinstance(module, nn.Linear):  # 如果子模块是线性层。
                nn.init.trunc_normal_(module.weight, std=0.02)  # 用截断正态分布初始化权重。
                if module.bias is not None:  # 如果线性层带有偏置项。
                    nn.init.zeros_(module.bias)  # 把偏置初始化为 0。
            elif isinstance(module, nn.Conv2d):  # 如果子模块是卷积层。
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")  # 使用 Kaiming 初始化卷积权重。
                if module.bias is not None:  # 如果卷积层带有偏置项。
                    nn.init.zeros_(module.bias)  # 把偏置初始化为 0。
            elif isinstance(module, nn.LayerNorm):  # 如果子模块是 LayerNorm。
                nn.init.ones_(module.weight)  # 把缩放参数初始化为 1。
                nn.init.zeros_(module.bias)  # 把偏移参数初始化为 0。

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 定义完整模型的前向传播流程。
        x = self.patch_embed(x)  # 第一步，把输入图像切成 patch 并映射成 token 序列。

        batch_size = x.shape[0]  # 取出当前批次大小，后面扩展 cls token 要用。
        cls_token = self.cls_token.expand(batch_size, -1, -1)  # 沿 batch 维复制 cls token，但不实际拷贝数据。
        x = torch.cat((cls_token, x), dim=1)  # 把 cls token 拼接到 patch token 序列最前面。

        x = x + self.pos_embed  # 为每个 token 加上可学习的位置编码。
        x = self.pos_drop(x)  # 对加完位置编码的序列施加 dropout。

        for block in self.blocks:  # 依次通过每一个 Transformer block。
            x = block(x)  # 更新当前层的 token 表示。

        x = self.norm(x)  # 对最终输出的所有 token 做归一化。
        cls_output = x[:, 0]  # 取出第 0 个 token，也就是 cls token 的输出表示。
        logits = self.head(cls_output)  # 通过线性分类头得到各类别 logits。
        return logits  # 返回最终分类结果。


if __name__ == "__main__":  # 只有直接运行当前文件时，下面的测试代码才会执行。
    model = VisionTransformer()  # 实例化一个默认配置的 Vision Transformer 模型。
    dummy = torch.randn(2, 1, 28, 28)  # 构造一批假的输入图像，形状为 2x1x28x28。
    logits = model(dummy)  # 把假数据送入模型，执行一次前向传播。
    print(logits.shape)  # 打印输出形状，正常情况下应为 [2, 10]。
