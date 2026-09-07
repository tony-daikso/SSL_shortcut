"""E5a/c：DINO multi-crop 資料增強管線。

標準版（`DINOMultiCropAugmentation`）照官方 DINO 配方：2 個 global crop
（224px，scale 0.4-1.0）+ N 個 local crop（96px，scale 0.05-0.4），各自套用
color jitter / grayscale / Gaussian blur / solarize 的隨機組合。

E5c 需要「用 E4 驗證過的新 augmentation 配方重訓」，這裡的
`add_noise_whitening=True` 選項會在標準配方之後，額外疊加
`e4_noise_whitening.whiten_noise`（E4 驗證過、能把殘留的感測器雜訊類指紋壓到
chance 以下、病理代價幾乎為零的手段）——每次呼叫用隨機 seed（訓練要的是隨機性，
不是 E3/E4 離線分析那種可重現的固定 seed）。
"""

import random

from PIL import Image, ImageFilter, ImageOps
from torchvision import transforms

from e4_noise_whitening import whiten_noise

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

GLOBAL_CROP_SIZE = 224
LOCAL_CROP_SIZE = 96
GLOBAL_SCALE = (0.4, 1.0)
LOCAL_SCALE = (0.05, 0.4)

# E4 驗證過的雜訊白化強度：results/e4_whitening_report.md 顯示 strength=1.0 時
# pattern_noise 降到 chance 以下、polyp_label 只掉 0.9 個百分點，是目前驗證過
# 代價最小、效果最明確的設定，E5c 預設沿用同一個值。
WHITENING_STRENGTH = 1.0


class GaussianBlur:
    def __init__(self, p: float, radius_range=(0.1, 2.0)):
        self.p = p
        self.radius_range = radius_range

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        radius = random.uniform(*self.radius_range)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))


class Solarize:
    def __init__(self, p: float, threshold: int = 128):
        self.p = p
        self.threshold = threshold

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        return ImageOps.solarize(img, threshold=self.threshold)


class NoiseWhitening:
    """套用 E4a 驗證過的雜訊白化，訓練時每次呼叫用隨機 seed（不追求可重現性，
    追求的是每個 crop 各自獨立的雜訊，這正是白化要達成的效果）。"""

    def __init__(self, strength: float = WHITENING_STRENGTH):
        self.strength = strength

    def __call__(self, img: Image.Image) -> Image.Image:
        seed = random.randint(0, 2**31 - 1)
        return whiten_noise(img, strength=self.strength, seed=seed)


def _color_distortion() -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8
        ),
        transforms.RandomGrayscale(p=0.2),
    ])


def _normalize() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class DINOMultiCropAugmentation:
    def __init__(self, n_local_crops: int = 6, add_noise_whitening: bool = False):
        self.n_local_crops = n_local_crops
        whitening = [NoiseWhitening()] if add_noise_whitening else []

        flip_and_color = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            _color_distortion(),
        ])

        self.global_crop_1 = transforms.Compose([
            transforms.RandomResizedCrop(GLOBAL_CROP_SIZE, scale=GLOBAL_SCALE, interpolation=Image.BICUBIC),
            flip_and_color,
            GaussianBlur(p=1.0),
            *whitening,
            _normalize(),
        ])
        self.global_crop_2 = transforms.Compose([
            transforms.RandomResizedCrop(GLOBAL_CROP_SIZE, scale=GLOBAL_SCALE, interpolation=Image.BICUBIC),
            flip_and_color,
            GaussianBlur(p=0.1),
            Solarize(p=0.2),
            *whitening,
            _normalize(),
        ])
        self.local_crop = transforms.Compose([
            transforms.RandomResizedCrop(LOCAL_CROP_SIZE, scale=LOCAL_SCALE, interpolation=Image.BICUBIC),
            flip_and_color,
            GaussianBlur(p=0.5),
            *whitening,
            _normalize(),
        ])

    def __call__(self, image: Image.Image) -> list:
        crops = [self.global_crop_1(image), self.global_crop_2(image)]
        crops += [self.local_crop(image) for _ in range(self.n_local_crops)]
        return crops
