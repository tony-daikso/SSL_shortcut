"""E3：augmentation 劑量反應共用工具。

計畫 §6 E3 的機制性論證：augmentation 消除捷徑的機制是讓該特徵在 crop 間變得
不可預測，但它只能作用在自己有參數化的維度上。color jitter 是色彩空間的全局
仿射變換（brightness/contrast/saturation 是純量縮放、hue 是色相旋轉），不改變
雜訊的空間相關結構、去馬賽克偽影排列、壓縮區塊邊界位置。

E0.5c 已經在「單一強度」下驗證了這個機制（見 e05_fingerprints.py、
results/e05_report.md）：pattern_noise（無色高頻雜訊，模擬感測器 PRNU）幾乎不受
color jitter 影響，color_shift（固定色相旋轉，模擬白平衡系統性色偏）被壓到接近
chance。E3 把「有/無 jitter」這個二元判斷換成「jitter 強度連續變化」，看兩條曲線
是否呈現「一條降一條平」的形狀，並加上第三條曲線（真實 polyp_label 可分性）當
「augmentation 有沒有連帶傷害到診斷訊號」的直接量測。

強度用一個乘數 `strength` 縮放 11_e05_prepare_images.py 已經校準過的基準 color
jitter 參數（brightness=contrast=saturation=0.2、hue=0.5）；hue 在 torchvision
的合法範圍是 [-0.5, 0.5]，強度 >1 時用 min(0.5, ...) 封頂，其餘三個參數線性放大。
strength=0 时退化成 identity transform（完全不套用 jitter），對應 E0.5 已有的
「無 jitter」基準點，可以直接跟 e05_report.md 的數字對照。
"""

import random

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import uniform_filter
from torchvision import transforms

BASE_BRIGHTNESS = BASE_CONTRAST = BASE_SATURATION = 0.2
BASE_HUE = 0.5

JITTER_STRENGTHS = [0.0, 0.5, 1.0, 1.5, 2.0]


def make_jitter(strength: float) -> transforms.ColorJitter:
    return transforms.ColorJitter(
        brightness=BASE_BRIGHTNESS * strength,
        contrast=BASE_CONTRAST * strength,
        saturation=BASE_SATURATION * strength,
        hue=min(0.5, BASE_HUE * strength),
    )


def jitter_with_seed(image: Image.Image, seed: int, strength: float) -> Image.Image:
    """固定 per-image seed 套用指定強度的 color jitter，跟
    11_e05_prepare_images.py 的 jitter_with_seed 同一個套路，保證重跑可重現。
    strength=0 時所有參數都是 0，ColorJitter 退化成 identity，直接回傳原圖等價的
    結果（仍走一次 transform 只是為了程式路徑一致，不是為了效能）。"""
    state = torch.get_rng_state()
    py_state = random.getstate()
    torch.manual_seed(seed)
    random.seed(seed)
    out = make_jitter(strength)(image)
    torch.set_rng_state(state)
    random.setstate(py_state)
    return out


def simple_ssim(img_a: Image.Image, img_b: Image.Image, win_size: int = 7) -> float:
    """輕量 SSIM 實作（灰階、單一尺度），只用來當 E3c 的「影像有沒有被整個毀掉」
    trivial-destruction control，不追求跟 scikit-image 逐位元一致——專案目前的
    依賴（pandas/numpy/pillow/scikit-learn/torch/scipy）已經足夠算這個，不需要
    額外裝 scikit-image。用 box filter（uniform_filter）近似高斯窗口局部統計量，
    標準 SSIM 公式（Wang et al. 2004）在 8-bit 影像上的慣用常數 C1/C2。"""
    a = np.asarray(img_a.convert("L"), dtype=np.float64)
    b = np.asarray(img_b.convert("L"), dtype=np.float64)
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    mu_a = uniform_filter(a, win_size)
    mu_b = uniform_filter(b, win_size)
    mu_a2, mu_b2, mu_ab = mu_a ** 2, mu_b ** 2, mu_a * mu_b

    sigma_a2 = uniform_filter(a * a, win_size) - mu_a2
    sigma_b2 = uniform_filter(b * b, win_size) - mu_b2
    sigma_ab = uniform_filter(a * b, win_size) - mu_ab

    ssim_map = ((2 * mu_ab + C1) * (2 * sigma_ab + C2)) / ((mu_a2 + mu_b2 + C1) * (sigma_a2 + sigma_b2 + C2))
    return float(ssim_map.mean())
