"""E3b（第三條曲線：病理可分性）：對全部 60 支影片、6000 張真實抽樣影格套用
`fov_protocol.unify_crop`，在 e3_augmentation_common.JITTER_STRENGTHS 每個強度下
套用 color jitter，跑 frozen DINOv2 抽 embedding。

跟 17_e3_extract_synthetic_embeddings.py 的兩條合成曲線不同，這條曲線用全資料集
規模（6000 張、非 500 張 pilot 底圖）——因為這是 E3 裡臨床上最關鍵的問題（真實
polyp 訊號有沒有被 augmentation 連帶壓下去），值得用跟 E1d 一樣的統計效力去量測，
不需要像合成校準實驗那樣受限於 pilot 規模。

輸出 results/e3_real_embeddings.npz：embeddings（N x 384）+ 對應的
frame_id/video_id/polyp_label/strength。
"""

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from config import REPO_ROOT, RESULTS_DIR
from fov_protocol import unify_crop
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

    frame_labels = pd.read_csv(RESULTS_DIR / "pilot_frame_labels.csv", dtype={"cohort": str})
    print(f"真實影格：{len(frame_labels)} 張（{frame_labels['video_id'].nunique()} 支影片）")

    print("套用 unify_crop 前處理（跟 E1 一致）...")
    cropped = [unify_crop(Image.open(REPO_ROOT / p)) for p in frame_labels["frame_path"]]

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
        imgs = [jitter_with_seed(im, seed=i, strength=strength) for i, im in enumerate(cropped)]
        for i in range(len(imgs)):
            r = frame_labels.iloc[i]
            rows.append({
                "frame_id": r["frame_id"], "video_id": r["video_id"],
                "polyp_label": r["polyp_label"], "strength": strength,
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

    out_path = RESULTS_DIR / "e3_real_embeddings.npz"
    np.savez(
        out_path,
        embeddings=embeddings,
        frame_id=df["frame_id"].values,
        video_id=df["video_id"].values,
        polyp_label=df["polyp_label"].values,
        strength=df["strength"].values,
    )
    print(f"Wrote {embeddings.shape} -> {out_path}")


if __name__ == "__main__":
    main()
