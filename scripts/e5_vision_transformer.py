"""E5a：極簡 ViT-S/16 實作，支援任意輸入解析度（位置編碼插值）。

DINO 的 multi-crop 訓練需要同一個 backbone 同時吃 224px 的 global crop 跟 96px
的 local crop——torchvision 內建的 `VisionTransformer` 建構時綁死單一
`image_size`，位置編碼無法動態插值，不適用。timm 有支援但這台機器裝不上（pip
被系統環境擋住），而且遠端 GPU server 不一定有裝 timm。

這裡照 DINO 論文/官方實作的標準 ViT-S 架構自己刻一份最小實作（patch embed →
CLS token + 位置編碼 → transformer blocks → 最終 LayerNorm），只依賴 torch，
確保跟現有 requirements.txt 相容、可攜到任何環境。

ViT-S 標準配置（跟 DINO/DINOv2 論文一致）：embed_dim=384, depth=12, num_heads=6,
mlp_ratio=4。patch_size=16（計畫 §6 E5a 明確要求 ViT-S/16，區別於 E1 用的
DINOv2 官方權重 ViT-S/14）。
"""

import math

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # B, embed_dim, H/patch, W/patch
        return x.flatten(2).transpose(1, 2)  # B, N, embed_dim


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return self.proj_drop(x)


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, qkv_bias: bool = True, drop: float = 0.0, attn_drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """輸出 CLS token 特徵（`embed_dim` 維），不含分類頭——DINO 的 projection
    head（見 e5_dino_model.py）會接在這之後。"""

    def __init__(
        self,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        base_img_size: int = 224,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(patch_size, in_chans, embed_dim)
        n_base_patches = (base_img_size // patch_size) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_base_patches + 1, embed_dim))

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def interpolate_pos_encoding(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """輸入解析度跟建構時的 base_img_size 不同時（DINO multi-crop 的 local
        crop 通常比 global crop 小很多），雙三次插值位置編碼到對應的 patch
        數量——這是 multi-crop 訓練能跑在同一個 backbone 上的關鍵。"""
        n_patches = x.shape[1] - 1
        n_pos = self.pos_embed.shape[1] - 1
        if n_patches == n_pos and h == w:
            return self.pos_embed

        cls_pos = self.pos_embed[:, :1]
        patch_pos = self.pos_embed[:, 1:]
        dim = patch_pos.shape[-1]
        orig_grid = int(math.sqrt(n_pos))
        new_h, new_w = h // self.patch_size, w // self.patch_size

        patch_pos = patch_pos.reshape(1, orig_grid, orig_grid, dim).permute(0, 3, 1, 2)
        patch_pos = nn.functional.interpolate(patch_pos, size=(new_h, new_w), mode="bicubic", align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, new_h * new_w, dim)
        return torch.cat([cls_pos, patch_pos], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.interpolate_pos_encoding(x, H, W)

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x[:, 0]  # CLS token


def vit_small(patch_size: int = 16) -> VisionTransformer:
    return VisionTransformer(patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.0)
