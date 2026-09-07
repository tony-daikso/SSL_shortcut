"""E3a/b（合成指紋兩條曲線）：對 E0.5 已經產生的 pattern_noise / color_shift 底圖
（`data/e05_synthetic/`，500 張、無 jitter），在 e3_augmentation_common.JITTER_STRENGTHS
每個強度下套用 color jitter，跑 frozen DINOv2 抽 embedding。

直接重用 E0.5 已經注入好指紋的底圖（不重新呼叫 apply_pattern_noise/apply_color_shift），
只在這裡疊加不同強度的 jitter——這樣兩條曲線在 strength=0 這一點的數字，理論上應該
跟 `results/e05_report.md` 的 pattern_noise/color_shift（無 jitter）數字一致，可以直接
拿來互相校驗腳本有沒有寫對。

輸出 results/e3_synthetic_embeddings.npz：embeddings（N x 384）+ 對應的
frame_id/video_id/synthetic_class/mechanism（pattern_noise 或 color_shift）/strength。
"""

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from config import REPO_ROOT, RESULTS_DIR
from e3_augmentation_common import JITTER_STRENGTHS, jitter_with_seed

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
    for strength in JITTER_STRENGTHS:
        print(f"=== strength={strength} ===")
        imgs = []
        for i, r in base.iterrows():
            src = Image.open(REPO_ROOT / r["image_path"]).convert("RGB")
            jittered = jitter_with_seed(src, seed=i, strength=strength)
            imgs.append(jittered)
            rows.append({
                "frame_id": r["frame_id"], "video_id": r["video_id"],
                "synthetic_class": r["synthetic_class"], "mechanism": r["variant"],
                "strength": strength,
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

    out_path = RESULTS_DIR / "e3_synthetic_embeddings.npz"
    np.savez(
        out_path,
        embeddings=embeddings,
        frame_id=df["frame_id"].values,
        video_id=df["video_id"].values,
        synthetic_class=df["synthetic_class"].values,
        mechanism=df["mechanism"].values,
        strength=df["strength"].values,
    )
    print(f"Wrote {embeddings.shape} -> {out_path}")


if __name__ == "__main__":
    main()
