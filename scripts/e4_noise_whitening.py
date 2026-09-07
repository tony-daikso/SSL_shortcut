"""E4a：雜訊重合成 / 雜訊白化。

E3 已經證實 color jitter（色彩空間的全局仿射變換）機制上動不到 pattern_noise
這類無色高頻雜訊的空間結構——不管 jitter 開多強，pattern_noise probe 都維持在
遠高於 chance 的殘餘地板。E4a 要測的是：一個**直接針對雜訊本身**設計的 augmentation
（而不是色彩空間的變換），能不能把這條原本壓不下去的曲線也壓下去。

真實感測器雜訊（PRNU）之所以能被 probe 讀出來，關鍵性質是「同一台相機拍的每一張
影格，雜訊的空間結構都高度相似（同一組固定的像素級增益圖樣）」。這裡的合成
pattern_noise 也是同樣的設計：同一個 synthetic_class 的每一張圖，都加了同一張
固定產生的雜訊紋理（見 e05_fingerprints.py 的 `_class_seed`）。

雜訊白化的做法：估計每張圖的高頻殘差（原圖 - 輕度模糊後的低頻部分），把這個殘差
按強度比例替換成**每張圖各自獨立抽樣**的新雜訊——這樣同一個 class 裡的每張圖就不再
共享同一組雜訊圖樣，probe 沒有「固定指紋」可以認。強度 0 時完全不改變原圖（保留
原本的殘差）；強度 1 時殘差被完全替換成新的獨立隨機雜訊。

跟 color jitter 這種「色彩空間仿射變換」不同，這個操作直接作用在雜訊所在的頻域/
空間結構上，是機制推導出來的針對性修法，不是通用配方。
"""

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

BLUR_SIGMA = 1.0
WHITENING_STRENGTHS = [0.0, 0.25, 0.5, 0.75, 1.0]

# E4 測試「在既有 augmentation 之上加上雜訊白化」，jitter 固定在 E3 網格裡的
# 基準強度（1.0 = 跟 E0.5/E1 用的 COLOR_JITTER 完全一樣的強度），不再重新 sweep
# jitter 強度——E3 已經把 jitter 這個維度掃過了。
REFERENCE_JITTER_STRENGTH = 1.0


def whiten_noise(image: Image.Image, strength: float, seed: int, blur_sigma: float = BLUR_SIGMA) -> Image.Image:
    """strength=0 精確還原原圖（模糊+殘差重組是恆等操作）；strength=1 殘差完全
    替換成每張圖各自獨立抽樣的新雜訊，兩者之間線性內插。"""
    arr = np.asarray(image.convert("RGB")).astype(np.float32)
    blurred = gaussian_filter(arr, sigma=(blur_sigma, blur_sigma, 0))
    residual = arr - blurred

    rng = np.random.RandomState(seed)
    fresh_noise = rng.normal(loc=0.0, scale=float(residual.std()), size=arr.shape).astype(np.float32)

    mixed_residual = (1.0 - strength) * residual + strength * fresh_noise
    out = np.clip(blurred + mixed_residual, 0, 255).astype(np.uint8)
    return Image.fromarray(out)
