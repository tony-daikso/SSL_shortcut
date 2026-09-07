"""E3c：trivial-destruction control——用 SSIM 確認曲線的變化不是單純「影像被整個
毀掉」造成的。如果 jitter 強度大到讓整張影像面目全非，那麼不管訊號是什麼種類，
probe accuracy 掉下去都不能算是「機制性選擇壓下某一類指紋」的證據，只是「圖爛了
什麼都讀不到」。

用 E0.5 的 500 張 clean 底圖（跟 17/18 兩支抽 embedding 腳本用的是同一批底圖來源），
對每個 e3_augmentation_common.JITTER_STRENGTHS 強度算 SSIM(clean, jittered)，
如果在計畫用的強度範圍內 SSIM 都還維持在高值（例如 >0.7），就能排除「trivial
destruction」這個混淆解釋。
"""

import numpy as np
import pandas as pd
from PIL import Image

from config import REPO_ROOT, RESULTS_DIR
from e3_augmentation_common import JITTER_STRENGTHS, jitter_with_seed, simple_ssim


def main():
    manifest = pd.read_csv(RESULTS_DIR / "e05_image_manifest.csv")
    clean = manifest[manifest["variant"] == "clean"].reset_index(drop=True)
    print(f"底圖：{len(clean)} 張 clean 影格")

    rows = []
    for i, r in clean.iterrows():
        src = Image.open(REPO_ROOT / r["image_path"]).convert("RGB")
        for strength in JITTER_STRENGTHS:
            jittered = jitter_with_seed(src, seed=i, strength=strength)
            score = simple_ssim(src, jittered)
            rows.append({"frame_id": r["frame_id"], "strength": strength, "ssim": score})

    df = pd.DataFrame(rows)
    out_csv = RESULTS_DIR / "e3_ssim_control.csv"
    df.to_csv(out_csv, index=False)

    summary = df.groupby("strength")["ssim"].agg(["mean", "std", "min", "max"])
    print(summary)
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
