"""E0.5 準備階段：把 pilot 的 500 張真實 REAL-Colon 影格當底圖，注入合成指紋，
產生 probe 要用的所有影像變體。

用真實內視鏡影格當底圖（而不是隨機自然影像），是為了讓合成指紋的可分性不要跟
「內容本身差異巨大」混在一起——真正要測的是 probe 能不能抓到疊加上去的合成訊號，
不是能不能分辨兩張完全不同的照片。

每張底圖（先套用 fov_protocol.unify_crop 裁成 224x224，跟 E1 之後實際會用的前處理
一致）產生 5 個變體：
- clean：只有裁切，沒有注入任何合成指紋（對照組）
- pattern_noise：注入低階/高頻的固定雜訊紋理
- pattern_noise_jitter：pattern_noise 之後再套用隨機 color jitter
- color_shift：注入高階/低頻的固定 RGB 色偏
- color_shift_jitter：color_shift 之後再套用隨機 color jitter

color jitter 用固定 per-image seed（不是每次都不同），保證這支腳本重跑結果可重現。
"""

import random

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from config import REPO_ROOT, RESULTS_DIR
from fov_protocol import unify_crop
from e05_fingerprints import N_CLASSES, apply_pattern_noise, apply_color_shift, assign_classes_balanced_per_video

OUT_DIR = REPO_ROOT / "data" / "e05_synthetic"
# 校準記錄（2026-09-03）：color_shift 已改成固定色相旋轉（見 e05_fingerprints.py），
# 跟 jitter 的 hue 參數直接對齊，所以 hue 開到上限（0.5）就足夠攻擊它；
# brightness/contrast/saturation 調低（0.2）是因為這幾個是純量縮放，會連帶
# 影響 pattern_noise 的絕對振幅、造成不該有的 collateral damage（先前用
# 0.6 時 pattern_noise 被壓過頭，掉了 21.6%，超過「應該幾乎不掉」的容忍範圍）。
COLOR_JITTER = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.5)


def jitter_with_seed(image: Image.Image, seed: int) -> Image.Image:
    state = torch.get_rng_state()
    py_state = random.getstate()
    torch.manual_seed(seed)
    random.seed(seed)
    out = COLOR_JITTER(image)
    torch.set_rng_state(state)
    random.setstate(py_state)
    return out


def main():
    frame_labels = pd.read_csv(RESULTS_DIR / "pilot_frame_labels.csv", dtype={"cohort": str})
    frame_labels["synthetic_class"] = assign_classes_balanced_per_video(frame_labels, N_CLASSES)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, r in frame_labels.iterrows():
        src_path = REPO_ROOT / r["frame_path"]
        base = unify_crop(Image.open(src_path))
        cls = int(r["synthetic_class"])

        variants = {
            "clean": base,
            "pattern_noise": apply_pattern_noise(base, cls),
            "color_shift": apply_color_shift(base, cls),
        }
        variants["pattern_noise_jitter"] = jitter_with_seed(variants["pattern_noise"], seed=i)
        variants["color_shift_jitter"] = jitter_with_seed(variants["color_shift"], seed=i)

        for variant_name, img in variants.items():
            out_path = OUT_DIR / variant_name / f"{r['frame_id']}.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(out_path)
            rows.append({
                "frame_id": r["frame_id"], "video_id": r["video_id"],
                "synthetic_class": cls, "variant": variant_name,
                "image_path": str(out_path.relative_to(REPO_ROOT)),
            })

    manifest = pd.DataFrame(rows)
    out_csv = RESULTS_DIR / "e05_image_manifest.csv"
    manifest.to_csv(out_csv, index=False)
    print(f"Wrote {len(manifest)} 列（{frame_labels.shape[0]} 張底圖 x 5 個變體）-> {out_csv}")
    print(frame_labels.groupby(["video_id", "synthetic_class"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
