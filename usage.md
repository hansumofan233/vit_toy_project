1.train

```bash
python train_vit.py
```

change parameters：

```bash
python train_vit.py --epochs 10 --batch-size 128 --learning-rate 3e-4
```

resume from checkpoint：

```bash
python train_vit.py --resume-checkpoint checkpoints/vit_fashionmnist_20260625_120000.pth
```

2.test
default：sample form test dataset

```bash
python test.py
```
default：newest checkpoint

```bash
python test.py --checkpoint checkpoints/vit_fashionmnist_20260628_021918.pth
```

test training dataset

```bash
python test.py --dataset train
```

Specify an index.

```bash
python test.py --index 12
```

3.batch test：successful rate

```bash
python batch_test.py
```

```bash
python batch_test.py --dataset train --num-samples 20 --checkpoint checkpoints/vit_fashionmnist.pth
```

1
x = model.patch_embed(image)
image.shape
image.dtype
image.device
2
x = self.proj(x)
x.shape
self.proj
self.patch_size
self.num_patches
3 4
x = x.flatten(2)
x = x.transpose(1, 2)
x.shape
5
x = torch.cat((cls_token, x), dim=1)
x.shape
cls_token.shape
model.cls_token.shape
F10 x.shape
6
x = x + model.pos_embed
x.shape
model.pos_embed.shape
F10 x.shape
7
x = x + self.attn(self.norm1(x))
x.shape
self.norm1
self.attn
F11 Step Into self.attn(self.norm1(x))
8
qkv = self.qkv(x)
x.shape
batch_size
num_tokens
embed_dim
self.num_heads
self.head_dim
self.scale
F10 qkv.shape
9
qkv = qkv.reshape(batch_size, num_tokens, 3, self.num_heads, self.head_dim)
qkv = qkv.permute(2, 0, 3, 1, 4)
qkv.shape
10
attn = (q @ k.transpose(-2, -1)) * self.scale
q.shape
k.shape
k.transpose(-2, -1).shape
self.scale
F10 attn.shape