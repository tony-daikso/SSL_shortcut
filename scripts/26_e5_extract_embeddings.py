"""E5b：從一個自訓 DINO checkpoint（25_e5_train_dino.py 的輸出）抽 embedding，
用跟 E1 完全一樣的真實影格集合（`pilot_frame_labels.csv`，全部 60 支影片、
6000 張）與前處理（unify_crop），確保直接可跟 `results/e1_embeddings.npz` 裡的
`dinov2_pretrained` / `dinov2_random` 放在同一組 probe 下比較，不是各自為政的
獨立量測。

用 teacher backbone（EMA 平均後的權重）抽 embedding——這是 DINO 訓練完後慣例
拿來用的版本，不是 student。
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from config import REPO_ROOT, RESULTS_DIR
from fov_protocol import unify_crop
from e5_vision_transformer import vit_small

BATCH_SIZE = 32
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="25_e5_train_dino.py 輸出的 checkpoint.pt 路徑")
    parser.add_argument("--tag", type=str, required=True,
                        help="這個 checkpoint 的簡短代號（例如 dino_selftrained_standard、"
                             "dino_selftrained_whitened），會出現在輸出檔名跟欄位名稱裡")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    backbone = vit_small(patch_size=16)
    backbone.load_state_dict(ckpt["teacher_backbone"])
    backbone.eval().to(device)
    print(f"讀取 checkpoint：{args.checkpoint}（訓練到 epoch {ckpt.get('epoch')}）")

    frame_labels = pd.read_csv(RESULTS_DIR / "pilot_frame_labels.csv", dtype={"cohort": str})
    print(f"真實影格：{len(frame_labels)} 張（{frame_labels['video_id'].nunique()} 支影片）")

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    print("套用 unify_crop 前處理（跟 E1 一致）...")
    cropped = [unify_crop(Image.open(REPO_ROOT / p)) for p in frame_labels["frame_path"]]

    embeddings = []
    for start in range(0, len(cropped), BATCH_SIZE):
        batch_imgs = cropped[start:start + BATCH_SIZE]
        batch_tensor = torch.stack([preprocess(im) for im in batch_imgs]).to(device)
        with torch.no_grad():
            feats = backbone(batch_tensor).cpu().numpy()
        embeddings.append(feats)
        print(f"  {min(start + BATCH_SIZE, len(cropped))}/{len(cropped)}", flush=True)

    embeddings = np.concatenate(embeddings, axis=0)

    out_path = RESULTS_DIR / f"e5_embeddings_{args.tag}.npz"
    np.savez(
        out_path,
        **{f"emb_{args.tag}": embeddings},
        frame_id=frame_labels["frame_id"].values,
        video_id=frame_labels["video_id"].values,
        cohort=frame_labels["cohort"].values,
        endoscope_brand=frame_labels["endoscope_brand"].values,
        polyp_label=frame_labels["polyp_label"].values,
    )
    print(f"Wrote {embeddings.shape} -> {out_path}")


if __name__ == "__main__":
    main()
