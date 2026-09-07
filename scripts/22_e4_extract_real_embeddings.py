"""E4d（第三條曲線：病理可分性，在雜訊白化下有沒有付出代價）：對全部 60 支影片、
6000 張真實抽樣影格套用 unify_crop → 固定強度 color jitter → 雜訊白化 sweep，
跑 frozen DINOv2 抽 embedding。

輸出 results/e4_real_embeddings.npz。
"""

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from config import REPO_ROOT, RESULTS_DIR
from fov_protocol import unify_crop
from e3_augmentation_common import jitter_with_seed
from e4_noise_whitening import WHITENING_STRENGTHS, REFERENCE_JITTER_STRENGTH, whiten_noise

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
    device = get_device()
    print(f"Using device: {device}")

    frame_labels = pd.read_csv(RESULTS_DIR / "pilot_frame_labels.csv", dtype={"cohort": str})
    print(f"真實影格：{len(frame_labels)} 張（{frame_labels['video_id'].nunique()} 支影片）")

    print("套用 unify_crop 前處理...")
    cropped = [unify_crop(Image.open(REPO_ROOT / p)) for p in frame_labels["frame_path"]]

    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False)
    model.eval().to(device)
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    rows = []
    all_embeddings = []
    for w_strength in WHITENING_STRENGTHS:
        print(f"=== whitening_strength={w_strength} ===")
        imgs = []
        for i, im in enumerate(cropped):
            jittered = jitter_with_seed(im, seed=i, strength=REFERENCE_JITTER_STRENGTH)
            whitened = whiten_noise(jittered, strength=w_strength, seed=i)
            imgs.append(whitened)
        for i in range(len(imgs)):
            r = frame_labels.iloc[i]
            rows.append({
                "frame_id": r["frame_id"], "video_id": r["video_id"],
                "polyp_label": r["polyp_label"], "whitening_strength": w_strength,
            })

        for start in range(0, len(imgs), BATCH_SIZE):
            batch = imgs[start:start + BATCH_SIZE]
            batch_tensor = torch.stack([preprocess(im) for im in batch]).to(device)
            with torch.no_grad():
                out = model.forward_features(batch_tensor)
                cls = out["x_norm_clstoken"].cpu().numpy()
            all_embeddings.append(cls)
            print(f"  {min(start + BATCH_SIZE, len(imgs))}/{len(imgs)}", flush=True)

    embeddings = np.concatenate(all_embeddings, axis=0)
    df = pd.DataFrame(rows)
    assert len(df) == embeddings.shape[0]

    out_path = RESULTS_DIR / "e4_real_embeddings.npz"
    np.savez(
        out_path,
        embeddings=embeddings,
        frame_id=df["frame_id"].values,
        video_id=df["video_id"].values,
        polyp_label=df["polyp_label"].values,
        whitening_strength=df["whitening_strength"].values,
    )
    print(f"Wrote {embeddings.shape} -> {out_path}")


if __name__ == "__main__":
    main()
