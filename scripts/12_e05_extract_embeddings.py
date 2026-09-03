"""E0.5：對 11_e05_prepare_images.py 產生的全部影像變體，跑 frozen DINOv2（ViT-S/14）
抽 CLS token embedding。這是 E1a 之後正式 probe 會用的同一個 backbone，E0.5 的整個
目的就是先在合成資料上驗證這條 pipeline（backbone → linear probe）本身有沒有效。

輸出 results/e05_embeddings.npz：embeddings（N x 384）+ 對應的 frame_id/video_id/
synthetic_class/variant，用 index 對齊。
"""

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from config import REPO_ROOT, RESULTS_DIR

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

    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False)
    model.eval().to(device)

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    manifest = pd.read_csv(RESULTS_DIR / "e05_image_manifest.csv")

    all_embeddings = []
    for start in range(0, len(manifest), BATCH_SIZE):
        batch = manifest.iloc[start:start + BATCH_SIZE]
        imgs = [preprocess(Image.open(REPO_ROOT / p).convert("RGB")) for p in batch["image_path"]]
        batch_tensor = torch.stack(imgs).to(device)
        with torch.no_grad():
            out = model.forward_features(batch_tensor)
            cls = out["x_norm_clstoken"].cpu().numpy()
        all_embeddings.append(cls)
        print(f"  {min(start + BATCH_SIZE, len(manifest))}/{len(manifest)}", flush=True)

    embeddings = np.concatenate(all_embeddings, axis=0)
    out_path = RESULTS_DIR / "e05_embeddings.npz"
    np.savez(
        out_path,
        embeddings=embeddings,
        frame_id=manifest["frame_id"].values,
        video_id=manifest["video_id"].values,
        synthetic_class=manifest["synthetic_class"].values,
        variant=manifest["variant"].values,
    )
    print(f"Wrote {embeddings.shape} -> {out_path}")


if __name__ == "__main__":
    main()
