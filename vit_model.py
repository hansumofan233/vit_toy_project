"""Small, readable Vision Transformer implementation for toy image classification.

The model is intentionally simple so the full image -> patch -> token -> attention
path is easy to inspect in a debugger.
"""

from __future__ import annotations

import torch
from torch import nn


class PatchEmbedding(nn.Module):
	def __init__(
		self,
		img_size: int = 28,
		patch_size: int = 7,
		in_channels: int = 1,
		embed_dim: int = 64,
	) -> None:
		super().__init__()
		if img_size % patch_size != 0:
			raise ValueError("img_size must be divisible by patch_size")

		self.img_size = img_size
		self.patch_size = patch_size
		self.num_patches = (img_size // patch_size) * (img_size // patch_size)

		self.proj = nn.Conv2d(
			in_channels=in_channels,
			out_channels=embed_dim,
			kernel_size=patch_size,
			stride=patch_size,
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.proj(x)
		x = x.flatten(2)
		x = x.transpose(1, 2)
		return x


class MultiHeadSelfAttention(nn.Module):
	def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
		super().__init__()
		if embed_dim % num_heads != 0:
			raise ValueError("embed_dim must be divisible by num_heads")

		self.embed_dim = embed_dim
		self.num_heads = num_heads
		self.head_dim = embed_dim // num_heads
		self.scale = self.head_dim ** -0.5

		self.qkv = nn.Linear(embed_dim, embed_dim * 3)
		self.attn_drop = nn.Dropout(dropout)
		self.proj = nn.Linear(embed_dim, embed_dim)
		self.proj_drop = nn.Dropout(dropout)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		batch_size, num_tokens, embed_dim = x.shape

		qkv = self.qkv(x)    #[1, 17, 192]
		qkv = qkv.reshape(batch_size, num_tokens, 3, self.num_heads, self.head_dim) #[1, 17, 3, 4, 16]
		qkv = qkv.permute(2, 0, 3, 1, 4)    #[3, 1, 4, 17, 16]
		q, k, v = qkv[0], qkv[1], qkv[2]

		attn = (q @ k.transpose(-2, -1)) * self.scale  #[1, 4, 17, 17]
		attn = attn.softmax(dim=-1)  #[1, 4, 17, 17]
		attn = self.attn_drop(attn)  #[1, 4, 17, 17]

		out = attn @ v  #[B, 4, 17, 16]
		out = out.transpose(1, 2).reshape(batch_size, num_tokens, embed_dim)  #[B, 17, 4, 16]-> [B, 17, 64]
		out = self.proj(out)
		out = self.proj_drop(out)
		return out


class FeedForward(nn.Module):
	def __init__(self, embed_dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(embed_dim, hidden_dim),
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Linear(hidden_dim, embed_dim),
			nn.Dropout(dropout),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.net(x)


class TransformerBlock(nn.Module):
	def __init__(
		self,
		embed_dim: int,
		num_heads: int,
		mlp_ratio: int = 4,
		dropout: float = 0.1,
	) -> None:
		super().__init__()
		self.norm1 = nn.LayerNorm(embed_dim)
		self.attn = MultiHeadSelfAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
		self.norm2 = nn.LayerNorm(embed_dim)
		self.mlp = FeedForward(embed_dim=embed_dim, hidden_dim=embed_dim * mlp_ratio, dropout=dropout)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = x + self.attn(self.norm1(x))
		x = x + self.mlp(self.norm2(x))
		return x


class VisionTransformer(nn.Module):
	def __init__(
		self,
		img_size: int = 28,
		patch_size: int = 7,
		in_channels: int = 1,
		num_classes: int = 10,
		embed_dim: int = 64,
		depth: int = 2,
		num_heads: int = 4,
		mlp_ratio: int = 4,
		dropout: float = 0.1,
	) -> None:
		super().__init__()

		self.patch_embed = PatchEmbedding(
			img_size=img_size,
			patch_size=patch_size,
			in_channels=in_channels,
			embed_dim=embed_dim,
		)
		num_patches = self.patch_embed.num_patches

		self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
		self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
		self.pos_drop = nn.Dropout(dropout)

		self.blocks = nn.ModuleList(
			[
				TransformerBlock(
					embed_dim=embed_dim,
					num_heads=num_heads,
					mlp_ratio=mlp_ratio,
					dropout=dropout,
				)
				for _ in range(depth)
			]
		)
		self.norm = nn.LayerNorm(embed_dim)
		self.head = nn.Linear(embed_dim, num_classes)

		self._init_weights()

	def _init_weights(self) -> None:
		nn.init.trunc_normal_(self.cls_token, std=0.02)
		nn.init.trunc_normal_(self.pos_embed, std=0.02)

		for module in self.modules():
			if isinstance(module, nn.Linear):
				nn.init.trunc_normal_(module.weight, std=0.02)
				if module.bias is not None:
					nn.init.zeros_(module.bias)
			elif isinstance(module, nn.Conv2d):
				nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
				if module.bias is not None:
					nn.init.zeros_(module.bias)
			elif isinstance(module, nn.LayerNorm):
				nn.init.ones_(module.weight)
				nn.init.zeros_(module.bias)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.patch_embed(x)

		batch_size = x.shape[0]
		cls_token = self.cls_token.expand(batch_size, -1, -1)
		x = torch.cat((cls_token, x), dim=1)
		x = x + self.pos_embed
		x = self.pos_drop(x)

		for block in self.blocks:
			x = block(x)

		x = self.norm(x)  #[1,17,64]
		cls_output = x[:, 0]   #[1, 64]
		logits = self.head(cls_output) #[1, 10]
		return logits


if __name__ == "__main__":
	model = VisionTransformer()
	dummy = torch.randn(2, 1, 28, 28)
	logits = model(dummy)
	print(logits.shape)
