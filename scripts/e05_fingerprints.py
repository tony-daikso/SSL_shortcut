"""E0.5：合成指紋注入邏輯。

兩種人工指紋，刻意設計成落在頻域的兩端，對應計畫 §2/E3 的機制性論證
（「augmentation 只能作用在它有參數化的維度上」）：

- **pattern_noise（低階/高頻）**：每個合成類別一個固定的隨機雜訊紋理（固定 seed
  產生一次，同一類別的每張圖都加上同一張雜訊圖），模擬感測器 PRNU 這類雜訊指紋。
  color jitter（色彩空間的全局仿射變換）不改變雜訊的空間相關結構，機制上不應該
  壓得下去。
- **color_shift（高階/低頻）**：每個合成類別一個固定的**色相（hue）旋轉角度**，
  模擬白平衡/色彩處理管線的系統性色偏。故意選色相旋轉而不是任意方向的 RGB 加法
  向量，是因為要跟 color jitter 的 hue 參數直接對齊（同一個參數空間），這樣
  E0.5c 的機制驗證才是公平的比較——不是「這個訊號本來就比較容易被 jitter 蓋掉」，
  而是「jitter 的參數空間本來就包含這個訊號的自由度」。

兩者都是「陽性對照」：訊號是人為注入、已知強度、已知有沒有被壓下去的正確答案，
用來檢驗 probe（frozen DINOv2 + linear probe）這個量測工具本身有沒有效
（E0.5a）、對兩種訊號的靈敏度是否對等（E0.5b），以及 augmentation 的機制性預測
在合成資料上成不成立（E0.5c）。
"""

import hashlib

import numpy as np
from PIL import Image

N_CLASSES = 4
IMAGE_SIZE = 224

# 強度用「等量測試」之前的初始猜測，E0.5b 會檢查兩者的 probe 靈敏度是否對等，
# 不對等的話再回頭調整這裡（見 13_e05_run_probes.py 的校準說明）。
PATTERN_NOISE_AMPLITUDE = 30.0  # 加到 0-255 尺度像素值上的雜訊標準差
COLOR_SHIFT_HUE_DEGREES = 40.0  # 固定色相旋轉角度（0-360 度）

# 校準記錄（2026-09-03，最終結果見 results/e05_report.md，三項判準全數通過）：
# 1. pattern_noise 振幅太小（18.0）：probe 只驗出 ~31% accuracy（chance 25%），
#    不滿足 E0.5a，加大振幅。
# 2. color_shift 一開始做成「任意方向的 RGB 加法向量」：振幅調到跟 pattern_noise
#    可分性相當後 E0.5a/b 都過，但 E0.5c 失敗——套用 color jitter 後 accuracy
#    只掉 ~3%（要求 >15%），因為 jitter 的 brightness/contrast/saturation/hue
#    這幾個參數不保證能抵銷一個任意方向的 RGB 向量。改成「固定色相旋轉」，直接
#    對齊 jitter 的 hue 自由度，才是公平的機制比較。
# 3. pattern_noise 一開始做成「三個通道各自獨立」的彩色雜訊：色相旋轉會打散通道
#    間的關係，等於意外污染到本來不該被 hue jitter 碰到的訊號，且 ViT 的 14x14
#    patch embedding 對這種高頻雜訊有類似平均池化的抑制效果，需要開很大的振幅
#    才追得上 color_shift 的可分性，而振幅一大又更容易被 jitter 的
#    brightness/contrast 順帶影響。改成「無色（同一個值同時加到三個通道）」後，
#    數學上 hue 旋轉完全動不到它（只轉色相、不改明度），振幅也不用開太大。
# 4. **關鍵 bug**：`_class_seed()` 一開始用 Python 內建 `hash((class_id, salt))`，
#    字串的 hash() 預設受 PYTHONHASHSEED 隨機化影響，每次重新執行這支腳本
#    （新的 Python process）都會得到不同的「固定」雜訊/色偏，完全不可重現——這曾
#    經讓校準過程中的數字大幅跳動、一度誤以為是振幅在起作用。改用 hashlib.md5
#    對固定編碼字串取雜湊後解決，兩次獨立執行 11_e05_prepare_images.py 產生的
#    color_shift 圖檔已驗證位元組對位元組相同。



def _class_seed(class_id: int, salt: str) -> int:
    """重要：不能用 Python 內建的 hash()——字串的 hash() 預設每個process 啟動時
    會用隨機的 PYTHONHASHSEED，同一組 (class_id, salt) 在不同次執行這支腳本時會
    得到不同的整數，等於每次重跑都換了一組「固定」雜訊/色偏，完全違背「固定 seed
    保證可重現」的設計初衷（這個 bug 曾經讓校準過程中的數字看起來不穩定，一度誤
    以為是雜訊振幅在起作用，其實是每次重跑都換了雜訊本身）。改用 hashlib 對固定
    編碼的位元組雜湊，結果不受 PYTHONHASHSEED 影響。"""
    digest = hashlib.md5(f"{class_id}:{salt}".encode()).digest()
    return int.from_bytes(digest[:4], "big") % (2**31)


def make_pattern_noise_texture(class_id: int, size: int = IMAGE_SIZE, amplitude: float = PATTERN_NOISE_AMPLITUDE) -> np.ndarray:
    """固定 seed 產生一張該類別專屬、之後每張圖都會加上同一張的雜訊紋理。

    刻意做成**無色（achromatic）**——同一個雜訊值同時加到 R/G/B 三個通道，只改變
    亮度、不改變色度。真實的感測器 PRNU 本質上是亮度域的固定增益圖樣，不是逐通道
    獨立的彩色雜訊；這裡若讓三個通道各自獨立加雜訊，色相旋轉（hue rotation）會把
    這個「彩色」雜訊的通道間關係打散，等於讓 color jitter 意外污染到本來不該碰到
    的訊號，機制驗證（E0.5c）就不公平了。做成無色之後，hue 旋轉在數學上完全不會
    動到它（HSV 的 H 分量旋轉不改變像素的明度 V），只有 pattern_noise 本身的空間
    結構會被 probe 讀到。"""
    rng = np.random.RandomState(_class_seed(class_id, "pattern_noise"))
    mono = rng.normal(loc=0.0, scale=amplitude, size=(size, size, 1))
    return np.repeat(mono, 3, axis=2)


def make_hue_shift_units(class_id: int, degrees: float = COLOR_SHIFT_HUE_DEGREES) -> int:
    """固定 seed 決定該類別的色相旋轉方向（正/負），角度大小固定，回傳 PIL "HSV"
    模式下 0-255 尺度的位移量（PIL 的 H 通道用 0-255 代表 0-360 度）。"""
    rng = np.random.RandomState(_class_seed(class_id, "color_shift"))
    sign = 1 if rng.rand() > 0.5 else -1
    return int(round(sign * degrees / 360.0 * 256))


def apply_pattern_noise(image: Image.Image, class_id: int) -> Image.Image:
    arr = np.asarray(image.convert("RGB")).astype(np.float32)
    noise = make_pattern_noise_texture(class_id, size=arr.shape[0])
    if noise.shape[:2] != arr.shape[:2]:
        # 理論上所有輸入圖都已經統一裁切成 IMAGE_SIZE，這裡防呆一下不同尺寸的情況
        noise = make_pattern_noise_texture(class_id, size=arr.shape[0])
    out = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def apply_color_shift(image: Image.Image, class_id: int) -> Image.Image:
    hsv = np.asarray(image.convert("HSV")).astype(np.int16)
    shift = make_hue_shift_units(class_id)
    hsv = hsv.copy()
    hsv[..., 0] = (hsv[..., 0] + shift) % 256
    return Image.fromarray(hsv.astype(np.uint8), mode="HSV").convert("RGB")


def assign_classes_balanced_per_video(df, n_classes: int = N_CLASSES, seed: int = 0):
    """每支來源影片內部用循環方式平均分配到各合成類別，確保每個合成類別裡都混有
    全部來源影片的內容——避免 probe 靠「認得出這是哪支真實影片」這個捷徑，而不是
    真的在讀注入的合成指紋。"""
    rng = np.random.RandomState(seed)
    labels = np.empty(len(df), dtype=int)
    for video_id, group in df.groupby("video_id"):
        idx = group.index.to_numpy().copy()
        rng.shuffle(idx)
        labels[df.index.get_indexer(idx)] = np.arange(len(idx)) % n_classes
    return labels
