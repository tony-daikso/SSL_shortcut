"""E1b：backbone 對照用的三種 frozen 特徵抽取器。

- **dinov2_pretrained**：官方預訓練權重（跟 E0.5 用的同一顆 ViT-S/14）。這是主角，
  E1a/c/d 都用它。
- **dinov2_random**：同樣的 ViT-S/14 架構，但權重隨機初始化、完全沒看過任何資料。
  這是計畫要求的重要下限——如果隨機初始化也能把 cohort/video ID 猜得很準，代表
  這些「指紋」強到光靠架構的歸納偏見（inductive bias）就讀得出來，不需要任何
  學習，論述必須改成「這是視覺 backbone 的通性，不是 SSL 目標函數學來的」。
- **imagenet_supervised**：ResNet-50，ImageNet 監督式訓練。跟 DINOv2 不是同一個
  架構家族（ViT vs CNN），這是已知的限制，比較時只能看「同樣是監督式視覺特徵，
  在完全不同的訓練目標/資料下，指紋可分性高不高」這個方向性問題，不是嚴格的
  對照實驗。
"""

import copy

import torch
from torchvision import models, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

PREPROCESS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _reset_parameters_recursive(module: torch.nn.Module, generator: torch.Generator):
    for m in module.modules():
        if hasattr(m, "reset_parameters"):
            with torch.random.fork_rng():
                torch.manual_seed(int(torch.randint(0, 2**31, (1,), generator=generator).item()))
                m.reset_parameters()


def load_dinov2_pretrained():
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False)
    return model.eval()


def load_dinov2_random():
    """跟 pretrained 版本架構完全相同，但把所有權重重新隨機初始化。"""
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False)
    model = copy.deepcopy(model)
    gen = torch.Generator().manual_seed(0)
    _reset_parameters_recursive(model, gen)
    return model.eval()


def load_resnet50_imagenet():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = torch.nn.Identity()  # 拿 2048-dim global average pooled 特徵，不做分類
    return model.eval()


BACKBONES = {
    "dinov2_pretrained": load_dinov2_pretrained,
    "dinov2_random": load_dinov2_random,
    "imagenet_supervised": load_resnet50_imagenet,
}


@torch.no_grad()
def extract_batch(model, backbone_name: str, batch_tensor: torch.Tensor) -> "np.ndarray":
    if backbone_name.startswith("dinov2"):
        out = model.forward_features(batch_tensor)
        return out["x_norm_clstoken"].cpu().numpy()
    return model(batch_tensor).cpu().numpy()
