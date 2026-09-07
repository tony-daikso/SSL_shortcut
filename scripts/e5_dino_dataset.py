"""E5a：REAL-Colon 訓練語料的資料集類別，套用跟 E0e/E1 一致的 unify_crop
前處理（先切掉 FOV 幾何線索，再讓 DINO 的 multi-crop 在這個已經幾何正規化的
224x224 影像上隨機取樣），避免自訓模型連 FOV 幾何這個已知捷徑都直接學進去。

`build_frame_pool()` 預設讀 `results/pilot_frame_labels.csv`（目前是全部 60 支
影片、每支影片中間 1/3 時間軸隨機抽的 100 張，見 config.py/09_pilot_sample_frames.py
的修正記錄）。E5a 原文要求「每支影片均勻抽樣，不用全部影格」——這批既有資料本身
就是每支影片抽樣後的結果，可以直接當訓練語料用；若要在遠端擴大到更多張/每支
影片、涵蓋更完整的時間軸，重新指定 `frame_labels_csv` 指向新的 manifest 即可，
不需要改這支腳本。
"""

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from config import REPO_ROOT, RESULTS_DIR
from fov_protocol import unify_crop


def build_frame_pool(frame_labels_csv: Path = None, max_frames_per_video: int = None, seed: int = 0) -> list:
    """回傳訓練用的影格路徑清單（絕對路徑字串）。`max_frames_per_video` 給小規模
    smoke test 用——從每支影片既有的抽樣結果裡再抽一個更小的子集，不重新下載/
    抽樣任何東西。"""
    csv_path = frame_labels_csv or (RESULTS_DIR / "pilot_frame_labels.csv")
    df = pd.read_csv(csv_path, dtype={"cohort": str})

    if max_frames_per_video is not None:
        df = (
            df.groupby("video_id", group_keys=False)
            .apply(lambda g: g.sample(n=min(max_frames_per_video, len(g)), random_state=seed))
        )

    paths = [str(REPO_ROOT / p) for p in df["frame_path"]]
    existing = [p for p in paths if Path(p).exists()]
    missing = len(paths) - len(existing)
    if missing:
        print(f"警告：{missing} 張影格路徑在本機找不到檔案（可能是原始影格已清除），已略過。")
    return existing


class DINOFrameDataset(Dataset):
    def __init__(self, frame_paths: list, multi_crop_transform):
        self.frame_paths = frame_paths
        self.multi_crop_transform = multi_crop_transform

    def __len__(self) -> int:
        return len(self.frame_paths)

    def __getitem__(self, idx: int) -> list:
        img = Image.open(self.frame_paths[idx]).convert("RGB")
        img = unify_crop(img)
        return self.multi_crop_transform(img)


def multi_crop_collate(batch: list) -> list:
    """DINOFrameDataset 每個樣本回傳「一張影格的多個 crop」（list of tensors，
    每個 crop 解析度可能不同）。collate 成：對每個 crop 位置，把整個 batch 的
    該位置 crop 疊成一個 tensor，最後回傳一個 list（長度 = 每張圖的 crop 數）。
    這正是 e5_dino_model.MultiCropWrapper 預期的輸入格式。"""
    n_crops = len(batch[0])
    return [torch.stack([sample[i] for sample in batch]) for i in range(n_crops)]
