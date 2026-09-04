"""E1a/b：對 pilot 的 500 張真實 REAL-Colon 影格（套用跟 E0e 一致的 unify_crop
前處理），跑三種 backbone（frozen DINOv2 預訓練 / DINOv2 隨機初始化 / ImageNet
監督式 ResNet-50）抽 embedding。

輸出 results/e1_embeddings.npz：每個 backbone 一組 embedding 陣列，用同一份
frame_id/video_id/cohort/endoscope_brand/polyp_label 對齊。
"""

import numpy as np
import pandas as pd
import torch
from PIL import Image

from config import REPO_ROOT, RESULTS_DIR
from fov_protocol import unify_crop
from e1_backbones import BACKBONES, PREPROCESS, get_device, extract_batch

BATCH_SIZE = 32


def main():
    device = get_device()
    print(f"Using device: {device}")

    frame_labels = pd.read_csv(RESULTS_DIR / "pilot_frame_labels.csv", dtype={"cohort": str})

    # 先把所有影格套用統一裁切協定，跟 E0e 之後、E1 之前的前處理一致，避免幾何
    # 線索（E0e 已經證實存在）汙染這裡要測的「內容/紋理層級」指紋可分性。
    print("套用 unify_crop 前處理...")
    cropped_images = [unify_crop(Image.open(REPO_ROOT / p)) for p in frame_labels["frame_path"]]

    results = {}
    for backbone_name, loader in BACKBONES.items():
        print(f"=== {backbone_name} ===")
        model = loader().to(device)
        embeddings = []
        for start in range(0, len(cropped_images), BATCH_SIZE):
            batch_imgs = cropped_images[start:start + BATCH_SIZE]
            batch_tensor = torch.stack([PREPROCESS(im) for im in batch_imgs]).to(device)
            embeddings.append(extract_batch(model, backbone_name, batch_tensor))
            print(f"  {min(start + BATCH_SIZE, len(cropped_images))}/{len(cropped_images)}", flush=True)
        results[backbone_name] = np.concatenate(embeddings, axis=0)
        del model

    out_path = RESULTS_DIR / "e1_embeddings.npz"
    np.savez(
        out_path,
        **{f"emb_{k}": v for k, v in results.items()},
        frame_id=frame_labels["frame_id"].values,
        video_id=frame_labels["video_id"].values,
        cohort=frame_labels["cohort"].values,
        endoscope_brand=frame_labels["endoscope_brand"].values,
        polyp_label=frame_labels["polyp_label"].values,
    )
    print(f"Wrote -> {out_path}")
    for k, v in results.items():
        print(f"  {k}: {v.shape}")


if __name__ == "__main__":
    main()
