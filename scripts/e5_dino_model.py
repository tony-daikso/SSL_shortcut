"""E5a：DINO 的 projection head、multi-crop wrapper、loss——照官方 DINO
（facebookresearch/dino）的架構設計，改寫成只依賴這個 repo 已有的依賴
（torch，不需要 timm/xformers）。

三個元件：
- `DINOHead`：backbone 輸出的 CLS token 特徵 → MLP → L2 normalize → 權重正規化
  （weight-normalized）的最後一層，投影到一個高維「原型」（prototype）空間。
  `norm_last_layer=True` 時凍結最後一層的權重範數（只學方向），這是官方實作裡
  穩定訓練前期的已知技巧。
- `MultiCropWrapper`：DINO 的 multi-crop 訓練會產生解析度不同的多組 crop（global
  224px、local 96px），這裡先把「相同解析度」的 crop 疊在一起各自過一次
  backbone（比每個 crop 各自呼叫一次更省），再把所有 crop 的輸出接回同一個
  DINOHead。
- `DINOLoss`：teacher 輸出先用一個跨批次動量更新的 center 做去偏、再用較低溫度
  的 softmax 銳化；student 輸出用較高溫度的 log_softmax。loss 是所有「student
  crop、teacher crop」配對的 cross-entropy，但排除同一個 crop 對自己的配對
  （teacher 只看 global crop，student 看全部 crop，這已經天然排除了大部分的
  自我配對，但兩個 global crop 之間仍要排除同一個 index 的自我配對）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        norm_last_layer: bool = True,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last_layer = nn.utils.parametrizations.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        self.last_layer.parametrizations.weight.original0.data.fill_(1)
        if norm_last_layer:
            self.last_layer.parametrizations.weight.original0.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        return self.last_layer(x)


class MultiCropWrapper(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, crops: list) -> torch.Tensor:
        # 依解析度分組：同一組裡的 crop shape 完全一樣，可以疊成一個大 batch
        # 一次過 backbone（DINO multi-crop 常見的效能技巧）。crops 不一定照
        # 解析度排序，直接用 dict 分組最穩健。
        groups = {}
        for i, c in enumerate(crops):
            groups.setdefault(c.shape[-1], []).append(i)

        outputs = [None] * len(crops)
        for size, idxs in groups.items():
            batch = torch.cat([crops[i] for i in idxs], dim=0)
            feats = self.backbone(batch)
            splits = torch.split(feats, [crops[i].shape[0] for i in idxs])
            for i, feat in zip(idxs, splits):
                outputs[i] = feat

        all_feats = torch.cat(outputs, dim=0)
        return self.head(all_feats)


class DINOLoss(nn.Module):
    def __init__(
        self,
        out_dim: int,
        n_global_crops: int = 2,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.n_global_crops = n_global_crops
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_output: torch.Tensor, teacher_output: torch.Tensor, teacher_temp: float, n_crops: int) -> torch.Tensor:
        student_out = student_output / self.student_temp
        student_chunks = student_out.chunk(n_crops)

        teacher_out = F.softmax((teacher_output - self.center) / teacher_temp, dim=-1)
        teacher_chunks = teacher_out.detach().chunk(self.n_global_crops)

        total_loss = 0.0
        n_terms = 0
        for t_idx, t_chunk in enumerate(teacher_chunks):
            for s_idx, s_chunk in enumerate(student_chunks):
                if s_idx == t_idx:
                    continue  # 同一個 crop 對自己不算 loss（避免平凡解）
                loss = torch.sum(-t_chunk * F.log_softmax(s_chunk, dim=-1), dim=-1)
                total_loss += loss.mean()
                n_terms += 1

        self.update_center(teacher_output)
        return total_loss / n_terms

    @torch.no_grad()
    def update_center(self, teacher_output: torch.Tensor):
        batch_center = teacher_output.mean(dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
