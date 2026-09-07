"""E4：整合三條曲線 + SSIM 對照，產生雜訊白化的驗證報告。

驗證迴路（計畫 §6 E4）：在 E3 已經固定的標準 jitter 強度之上，額外 sweep 雜訊白化
強度，看 E3 裡壓不下去的 pattern_noise 曲線這次會不會開始下降；同時檢查
color_shift（預期應該維持在 jitter 已經壓過的水準，不再進一步下降，因為白化動的
是雜訊不是色彩）與 polyp_label（E4d，白化有沒有連帶傷害病理訊號）。
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from config import RESULTS_DIR

N_REPEATS = 10


def probe_by_group(X, y, groups, n_repeats=N_REPEATS):
    accs = []
    for seed in range(n_repeats):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train_idx, test_idx = next(gss.split(X, y, groups=groups))
        scaler = StandardScaler().fit(X[train_idx])
        clf = LogisticRegression(max_iter=2000)
        clf.fit(scaler.transform(X[train_idx]), y[train_idx])
        pred = clf.predict(scaler.transform(X[test_idx]))
        accs.append((pred == y[test_idx]).mean())
    accs = np.array(accs)
    return float(accs.mean()), float(accs.std())


def probe_held_out_frame(X, y, n_repeats=N_REPEATS):
    accs = []
    for seed in range(n_repeats):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train_idx, test_idx = next(sss.split(X, y))
        scaler = StandardScaler().fit(X[train_idx])
        clf = LogisticRegression(max_iter=2000)
        clf.fit(scaler.transform(X[train_idx]), y[train_idx])
        pred = clf.predict(scaler.transform(X[test_idx]))
        accs.append((pred == y[test_idx]).mean())
    accs = np.array(accs)
    return float(accs.mean()), float(accs.std())


def main():
    synth = np.load(RESULTS_DIR / "e4_synthetic_embeddings.npz", allow_pickle=True)
    real = np.load(RESULTS_DIR / "e4_real_embeddings.npz", allow_pickle=True)
    ssim_df = pd.read_csv(RESULTS_DIR / "e4_ssim_control.csv")

    synth_df = pd.DataFrame({
        "video_id": synth["video_id"], "synthetic_class": synth["synthetic_class"],
        "mechanism": synth["mechanism"], "whitening_strength": synth["whitening_strength"],
    })
    synth_emb = synth["embeddings"]

    real_df = pd.DataFrame({
        "video_id": real["video_id"], "polyp_label": real["polyp_label"],
        "whitening_strength": real["whitening_strength"],
    })
    real_emb = real["embeddings"]

    strengths = sorted(set(synth_df["whitening_strength"].tolist()))
    rows = []
    for w_strength in strengths:
        row = {"whitening_strength": w_strength}
        for mechanism in ["pattern_noise", "color_shift"]:
            mask = (synth_df["mechanism"] == mechanism) & (synth_df["whitening_strength"] == w_strength)
            X = synth_emb[mask.values]
            y = synth_df.loc[mask, "synthetic_class"].values
            groups = synth_df.loc[mask, "video_id"].values
            mean_acc, std_acc = probe_by_group(X, y, groups)
            row[f"{mechanism}_acc"] = mean_acc
            row[f"{mechanism}_std"] = std_acc

        mask = real_df["whitening_strength"] == w_strength
        X = real_emb[mask.values]
        y = real_df.loc[mask, "polyp_label"].values
        mean_acc, std_acc = probe_held_out_frame(X, y)
        row["polyp_acc"] = mean_acc
        row["polyp_std"] = std_acc

        ssim_row = ssim_df[ssim_df["whitening_strength"] == w_strength]["ssim"]
        row["ssim_mean"] = float(ssim_row.mean())

        rows.append(row)

    curve = pd.DataFrame(rows)

    chance_4way = 0.25
    polyp_majority = pd.Series(real_df["polyp_label"]).value_counts(normalize=True).max()

    # E3 在 strength=1.0（跟這裡的 REFERENCE_JITTER_STRENGTH 一致）的無白化基準，
    # 直接寫死引用 results/e3_dose_response_report.md 的數字，供對照。
    e3_pattern_noise_ref = 0.430
    e3_color_shift_ref = 0.235
    e3_polyp_ref = 0.8862

    lines = ["# E4a/d：雜訊白化驗證報告（全部 60 支影片 + E0.5 合成指紋，正式結論）\n"]
    lines.append(
        f"jitter 固定在 E3 的標準強度（strength=1.0，對照數字：pattern_noise "
        f"{e3_pattern_noise_ref:.3f}、color_shift {e3_color_shift_ref:.3f}、polyp_label "
        f"{e3_polyp_ref:.3f}，見 `e3_dose_response_report.md`），額外 sweep 雜訊白化強度 "
        "0.0-1.0（0=不白化，等同上面 E3 的基準；1=殘差完全替換成獨立隨機雜訊）。\n"
    )
    lines.append(curve.round(4).to_markdown(index=False))
    lines.append("")

    pn_start, pn_end = curve["pattern_noise_acc"].iloc[0], curve["pattern_noise_acc"].iloc[-1]
    cs_start, cs_end = curve["color_shift_acc"].iloc[0], curve["color_shift_acc"].iloc[-1]
    polyp_start, polyp_end = curve["polyp_acc"].iloc[0], curve["polyp_acc"].iloc[-1]
    pn_drop = pn_start - pn_end
    cs_drop = cs_start - cs_end
    polyp_drop = polyp_start - polyp_end

    lines.append("## 解讀\n")
    lines.append(
        f"**pattern_noise**：{pn_start:.3f} → {pn_end:.3f}（掉了 {pn_drop:.3f}，"
        f"E3 基準是 {e3_pattern_noise_ref:.3f}）。\n\n"
        f"**color_shift**：{cs_start:.3f} → {cs_end:.3f}（變化 {cs_drop:+.3f}，"
        f"E3 基準是 {e3_color_shift_ref:.3f}）。\n\n"
        f"**polyp_label**：{polyp_start:.3f} → {polyp_end:.3f}（掉了 {polyp_drop:.3f}，"
        f"majority baseline {polyp_majority:.3f}，E3 基準是 {e3_polyp_ref:.3f}）。\n\n"
        f"**SSIM**：whitening_strength=1.0 時均值 {curve['ssim_mean'].iloc[-1]:.3f}"
        f"（strength=0 時 {curve['ssim_mean'].iloc[0]:.3f}）。\n"
    )

    pn_meaningfully_dropped = pn_drop > 0.10 and pn_end < pn_start * 0.8
    cs_stayed_flat = abs(cs_drop) < 0.10
    if pn_meaningfully_dropped:
        verdict = (
            f"**驗證迴路成立**：加上雜訊白化後，E3 裡壓不下去的 pattern_noise 曲線"
            f"這次確實開始下降（{pn_start:.3f} → {pn_end:.3f}）。機制推導的修法"
            "（打亂雜訊的空間結構、換成每張圖獨立的隨機雜訊）有效——原本殘留的"
            "感測器雜訊類指紋，可以用針對性的手段壓下去，不是 color jitter 那類"
            "通用配方碰不到就永遠碰不到。"
        )
    else:
        verdict = (
            f"**驗證迴路未成立**：pattern_noise 在雜訊白化下沒有明顯下降"
            f"（{pn_start:.3f} → {pn_end:.3f}）。可能的原因：(a) 白化強度不夠、"
            "blur_sigma 沒抓對雜訊所在的頻段；(b) probe 讀到的其實不是雜訊本身的"
            "空間結構，而是白化操作本身殘留的某種副作用統計量；(c) 機制歸因本身"
            "有誤，需要回頭檢查。這本身也是有用的負面資訊，不代表白化完全無效，"
            "但這個特定實作沒有達到預期效果。"
        )
    lines.append(f"\n{verdict}\n")

    lines.append(
        f"\n**color_shift 的變化**：{'維持平穩，符合預期（白化動的是雜訊不是色彩）' if cs_stayed_flat else '出現超過 0.10 的變化，需要檢查白化操作是否意外影響了色彩統計量（例如高頻殘差抽取過程中是否洩漏了色彩資訊）'}。\n"
    )

    lines.append("## E4d：病理代價\n")
    if polyp_drop < 0.05:
        polyp_verdict = (
            f"**幾乎沒有代價**（掉了 {polyp_drop:.3f}）：雜訊白化在達到"
            f"{'壓制 pattern_noise 的效果' if pn_meaningfully_dropped else '目前的強度範圍'}"
            "的同時，沒有明顯傷害真實病理訊號。"
        )
    else:
        polyp_verdict = (
            f"**有實質代價**（掉了 {polyp_drop:.3f}，majority baseline "
            f"{polyp_majority:.3f}）：雜訊白化{'成功壓下 pattern_noise，但' if pn_meaningfully_dropped else '在還沒明顯壓下 pattern_noise 之前就'}"
            "已經開始損傷病理可分性，代表這個特定強度/實作方式的白化對診斷訊號不是"
            "免費的，需要找到「指紋降到可接受水準、病理代價還沒超過門檻」的中間強度，"
            "而不是直接用最大強度。"
        )
    lines.append(f"{polyp_verdict}\n")

    out_path = RESULTS_DIR / "e4_whitening_report.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(curve)


if __name__ == "__main__":
    main()
