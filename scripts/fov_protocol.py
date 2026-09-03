"""E0e-3：統一裁切協定。

## 修正記錄（2026-09-03）

最初的版本只做「中央方形裁切（取 min(width,height) 正方形）→ resize」，沒有額外
inset。後來發現這個版本不夠：E0d 原本用「整行/整列是否全黑」檢查 FOV 遮罩殘留，
結論是「幾乎沒有殘留」，但這個檢查方法本身有漏洞——內視鏡的 FOV 遮罩是圓形/八邊形，
黑色只出現在四個角落，不會讓「整行」或「整列」全黑，所以原本的檢查完全沒偵測到它。
肉眼複查 E0e-4 存的裁切前後對照圖時才發現：**幾乎每一張影格的四個角落都有明顯的黑色
遮罩**（在 200 張隨機抽樣影格上量測，20% 大小的角落框內平均有 40% 像素是黑的，
100% 的抽樣影格都有這個現象），比單純「影格寬高不同」更普遍、更系統性的幾何洩漏
來源。這個發現已經回頭訂正進 `results/dataset_self_check.md` 的勘誤區塊。

校準結果（見 `results/fov_crop_margin_calibration.md`，由 05_calibrate_crop_margin.py
產生）：在「中央方形裁切」之後，inset 從 0 加到 0.05 就能讓平均角落殘留從 ~11.4% 驟降
到 ~0.32%，之後（0.05 到 0.30）平均值只是緩慢下降、殘留影格比例在 3-8% 之間小幅
震盪（推測是少數影格本身邊緣有真實暗部內容——例如腸腔陰影，不是遮罩——inset 再大也
無法消除）。選 INSET_FRACTION = 0.15 當預設值：比最小可行值（0.05）保守一些，同時
避免為了追殺極少數離群影格而犧牲太多可用影像內容。

做法：
1. 中央方形裁切：取 min(width, height) 的正方形，從影格中心切。
2. 再往內裁掉 INSET_FRACTION 比例的邊距（四邊各裁掉 side * INSET_FRACTION）。
3. resize 到固定 TARGET_SIZE x TARGET_SIZE。

裁完之後每張影格的 width/height/aspect_ratio/area 全部變成同一個常數；角落遮罩
（corner_black_fraction，見下方 corner_black_fraction()）也被裁掉，兩者都不再帶有
跨影片的幾何變異，E0e-4 會實際重跑分類器驗證這件事，不是只憑理論保證。

TARGET_SIZE 選 224 只是一個通用預設值（常見 ImageNet 類 backbone 的輸入尺寸），跟
DINOv2 實際使用的輸入尺寸（例如 518）沒有綁定關係——E1 若用不同 backbone/輸入尺寸，
裁切邏輯不用改，只需要調整這個常數。
"""

import numpy as np
from PIL import Image

TARGET_SIZE = 224
INSET_FRACTION = 0.15


def unify_crop(image: Image.Image, target_size: int = TARGET_SIZE, inset_fraction: float = INSET_FRACTION) -> Image.Image:
    w, h = image.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    cropped = image.crop((left, top, left + side, top + side))

    inset = int(side * inset_fraction)
    if inset > 0:
        cropped = cropped.crop((inset, inset, side - inset, side - inset))

    return cropped.resize((target_size, target_size), Image.BILINEAR)


def corner_black_fraction(image: Image.Image, corner_frac: float = 0.15, thresh: int = 15) -> float:
    """四個角落框（每個框邊長 = corner_frac * min(width,height)）裡，多暗的像素比例
    （灰階強度 < thresh 視為黑）的平均值。用來量化 FOV 遮罩殘留，是 corner-only 的
    量測，不會像「整行/整列全黑」那種檢查一樣漏掉圓形/八邊形遮罩。"""
    arr = np.asarray(image.convert("L"))
    h, w = arr.shape
    ch = max(1, int(h * corner_frac))
    cw = max(1, int(w * corner_frac))
    corners = [arr[0:ch, 0:cw], arr[0:ch, w - cw:w], arr[h - ch:h, 0:cw], arr[h - ch:h, w - cw:w]]
    return float(np.mean([np.mean(c < thresh) for c in corners]))
