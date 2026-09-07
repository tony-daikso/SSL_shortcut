"""E4a（合成指紋兩條曲線，加上雜訊白化）：對 E0.5 的 pattern_noise / color_shift
底圖，先套用固定強度（e4_noise_whitening.REFERENCE_JITTER_STRENGTH，等同 E0.5/E1
用的標準 color jitter）的 color jitter，再疊加 e4_noise_whitening.WHITENING_STRENGTHS
每個強度的雜訊白化，跑 frozen DINOv2 抽 embedding。

跟 17_e3_extract_synthetic_embeddings.py 的差異：E3 是「單獨 sweep jitter 強度、
不加雜訊白化」；這裡是「jitter 固定在標準強度、額外 sweep 雜訊白化強度」——測試的是
「在既有 augmentation 之上加上這個新手段，能不能把 E3 壓不下去的 pattern_noise
曲線也壓下去」。

輸出 results/e4_synthetic_embeddings.npz。
"""

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from config import REPO_ROOT, RESULTS_DIR
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

    manifest = pd.read_csv(RESULTS_DIR / "e05_image_manifest.csv")
    base = manifest[manifest["variant"].isin(["pattern_noise", "color_shift"])].reset_index(drop=True)
    print(f"底圖：{len(base)} 張（{base['variant'].value_counts().to_dict()}）")

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
        for i, r in base.iterrows():
            src = Image.open(REPO_ROOT / r["image_path"]).convert("RGB")
            jittered = jitter_with_seed(src, seed=i, strength=REFERENCE_JITTER_STRENGTH)
            whitened = whiten_noise(jittered, strength=w_strength, seed=i)
            imgs.append(whitened)
            rows.append({
                "frame_id": r["frame_id"], "video_id": r["video_id"],
                "synthetic_class": r["synthetic_class"], "mechanism": r["variant"],
                "whitening_strength": w_strength,
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

    out_path = RESULTS_DIR / "e4_synthetic_embeddings.npz"
    np.savez(
        out_path,
        embeddings=embeddings,
        frame_id=df["frame_id"].values,
        video_id=df["video_id"].values,
        synthetic_class=df["synthetic_class"].values,
        mechanism=df["mechanism"].values,
        whitening_strength=df["whitening_strength"].values,
    )
    print(f"Wrote {embeddings.shape} -> {out_path}")


if __name__ == "__main__":
    main()
