"""E4：trivial-destruction control，跟 19_e3_ssim_control.py 同一套邏輯，換成量測
「固定強度 jitter + 雜訊白化 sweep」相對於原圖的 SSIM。雜訊白化直接操作高頻殘差，
預期比單純 color jitter 更容易讓 SSIM 掉得多，需要明確量出來，讓 E4d（病理代價）
的解讀有一個對照基準。
"""

import pandas as pd
from PIL import Image

from config import REPO_ROOT, RESULTS_DIR
from e3_augmentation_common import jitter_with_seed, simple_ssim
from e4_noise_whitening import WHITENING_STRENGTHS, REFERENCE_JITTER_STRENGTH, whiten_noise


def main():
    manifest = pd.read_csv(RESULTS_DIR / "e05_image_manifest.csv")
    clean = manifest[manifest["variant"] == "clean"].reset_index(drop=True)
    print(f"底圖：{len(clean)} 張 clean 影格")

    rows = []
    for i, r in clean.iterrows():
        src = Image.open(REPO_ROOT / r["image_path"]).convert("RGB")
        jittered = jitter_with_seed(src, seed=i, strength=REFERENCE_JITTER_STRENGTH)
        for w_strength in WHITENING_STRENGTHS:
            whitened = whiten_noise(jittered, strength=w_strength, seed=i)
            score = simple_ssim(src, whitened)
            rows.append({"frame_id": r["frame_id"], "whitening_strength": w_strength, "ssim": score})

    df = pd.DataFrame(rows)
    out_csv = RESULTS_DIR / "e4_ssim_control.csv"
    df.to_csv(out_csv, index=False)

    summary = df.groupby("whitening_strength")["ssim"].agg(["mean", "std", "min", "max"])
    print(summary)
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
