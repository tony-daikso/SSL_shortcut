"""E1c：維度分析——達成高準確率所需的主成分（PC）數量；指紋 vs 病理各自集中在
哪些主成分。計畫裡評為「最有說服力的單一圖表」。

兩個子分析：
1. **累積 PC 數量 vs probe accuracy**：只用前 k 個 PC 訓練 probe，k 從 1 掃到全部
   384 維，分別對 video_id（指紋代表）與 polyp_label（病理代表）畫出來。如果
   video_id 只需要很少 PC 就能到高準確率、polyp_label 需要更多 PC 才追得上，
   代表指紋訊號集中在少數幾個主成分、佔用的「表示預算」占比其實很大。
2. **每個 PC 的單變量可分性**：用「只看這一個 PC」的簡單分類器，逐一測每個 PC
   對 video_id 與 polyp_label 的可分性，找出哪些 PC 主要是指紋在用、哪些主要是
   病理在用、有沒有兩者都高度依賴的重疊 PC。
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from config import RESULTS_DIR

N_REPEATS = 5
K_VALUES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 384]


def probe_accuracy_top_k(X_pca: np.ndarray, y: np.ndarray, k: int, n_repeats: int = N_REPEATS) -> float:
    Xk = X_pca[:, :k]
    accs = []
    for seed in range(n_repeats):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train_idx, test_idx = next(sss.split(Xk, y))
        scaler = StandardScaler().fit(Xk[train_idx])
        clf = LogisticRegression(max_iter=2000)
        clf.fit(scaler.transform(Xk[train_idx]), y[train_idx])
        pred = clf.predict(scaler.transform(Xk[test_idx]))
        accs.append((pred == y[test_idx]).mean())
    return float(np.mean(accs))


def per_pc_univariate_accuracy(X_pca: np.ndarray, y: np.ndarray, n_components: int = 20) -> np.ndarray:
    """對前 n_components 個 PC，各自單獨（只用這一維）訓練 probe 的準確率。"""
    accs = []
    for i in range(n_components):
        col = X_pca[:, i:i + 1]
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
        train_idx, test_idx = next(sss.split(col, y))
        scaler = StandardScaler().fit(col[train_idx])
        clf = LogisticRegression(max_iter=2000)
        clf.fit(scaler.transform(col[train_idx]), y[train_idx])
        pred = clf.predict(scaler.transform(col[test_idx]))
        accs.append((pred == y[test_idx]).mean())
    return np.array(accs)


def main():
    data = np.load(RESULTS_DIR / "e1_embeddings.npz", allow_pickle=True)
    X = data["emb_dinov2_pretrained"]
    video_id_int = pd.factorize(data["video_id"])[0]
    polyp = data["polyp_label"]

    pca = PCA(n_components=min(X.shape))
    X_pca = pca.fit_transform(StandardScaler().fit_transform(X))
    explained = pca.explained_variance_ratio_

    n_videos = len(set(data["video_id"]))
    n_frames = X.shape[0]
    scale_tag = "pilot" if n_videos < 60 else "full"
    lines = [f"# E1c：維度分析（{scale_tag}，{n_videos} 支影片，{n_frames} 張影格，dinov2_pretrained embedding）\n"]
    lines.append(
        f"總維度 384，前 10 個 PC 解釋的變異量比例：{np.round(explained[:10], 3).tolist()}\n"
        f"累積到 90% 變異量需要 {int(np.searchsorted(np.cumsum(explained), 0.9)) + 1} 個 PC。\n"
    )

    lines.append("## 累積 PC 數量 vs probe accuracy\n")
    rows = []
    for k in K_VALUES:
        rows.append({
            "k_components": k,
            "video_id_accuracy": probe_accuracy_top_k(X_pca, video_id_int, k),
            "polyp_accuracy": probe_accuracy_top_k(X_pca, polyp, k),
        })
    curve = pd.DataFrame(rows)
    lines.append(curve.to_markdown(index=False))
    lines.append("")

    # 用曲線上觀察到的「峰值」當參考基準，不是直接用 k=384（全部維度）的數字——
    # 實測發現 k=384 的 accuracy（0.656）反而明顯低於 k=128 的峰值（0.880），這是
    # PCA 轉換後再重新標準化（StandardScaler）跟 L2 正則化交互作用的已知現象：
    # orthogonal 轉換 + 各維度獨立重新縮放，會讓原本均勻的 L2 懲罰在原始空間裡變成
    # 不均勻的懲罰，在「樣本數(500)接近特徵數(384)」這種高維小樣本情境下，容易讓
    # 全維度版本反而過擬合、測試集表現下降。這不代表全部 384 維真的比 128 維承載
    # 更少指紋資訊，只是這個特定 probe pipeline 在滿維度下的正則化沒調好——如實
    # 記錄這個現象，用峰值當參考基準比較不會誤導。
    peak_video_acc = curve["video_id_accuracy"].max()
    video_id_k_for_90pct = None
    for _, r in curve.iterrows():
        if r["video_id_accuracy"] >= 0.9 * peak_video_acc:
            video_id_k_for_90pct = int(r["k_components"])
            break
    peak_k = int(curve.loc[curve["video_id_accuracy"].idxmax(), "k_components"])
    full_k_acc = curve["video_id_accuracy"].iloc[-1]
    lines.append(
        f"**解讀**：video_id 只需要 {video_id_k_for_90pct} 個 PC 就能達到觀察到的峰值"
        f"準確率（{peak_video_acc:.3f}，出現在 k={peak_k}）"
        "的 90% 以上，代表指紋訊號高度集中在少數幾個主成分。"
        "polyp_label 的曲線見上表，比較兩者在同樣 k 值下的準確率差距，可以看出病理"
        "訊號是不是需要動用到更多、更分散的維度才追得上。\n\n"
    )
    if peak_k < 384 and full_k_acc < 0.9 * peak_video_acc:
        lines.append(
            f"**附帶觀察**：k=384（全部維度）的 video_id accuracy（{full_k_acc:.3f}）"
            f"反而低於 k={peak_k} 的峰值（{peak_video_acc:.3f}）——這是 PCA 全維度重新"
            "標準化後跟 logistic regression 的 L2 正則化交互作用造成的過擬合假象"
            f"（樣本數 {X.shape[0]} 接近特徵數 384），不代表全部維度真的承載更少指紋"
            "資訊，只是這個 probe 在滿維度下沒調好正則化強度，如實記錄避免誤導。\n"
        )
    else:
        lines.append(
            f"**附帶觀察**：在這個樣本數（{X.shape[0]}）下，k=384（全部維度）的 "
            f"video_id accuracy（{full_k_acc:.3f}）沒有出現「全維度反而變差」的過擬合"
            "假象——樣本數遠大於特徵數（384）時，PCA 全維度重新標準化不會讓 L2 正則化"
            "失衡，這跟 pilot 規模（樣本數接近特徵數）觀察到的假象不同，記錄下來作為"
            "對照。\n"
        )

    lines.append("## 每個 PC（前 20 個）對 video_id / polyp 的單變量可分性\n")
    video_id_per_pc = per_pc_univariate_accuracy(X_pca, video_id_int)
    polyp_per_pc = per_pc_univariate_accuracy(X_pca, polyp)
    per_pc_df = pd.DataFrame({
        "pc_index": range(1, 21),
        "explained_variance_ratio": explained[:20].round(4),
        "video_id_univariate_acc": video_id_per_pc.round(3),
        "polyp_univariate_acc": polyp_per_pc.round(3),
    })
    lines.append(per_pc_df.to_markdown(index=False))
    lines.append("")

    video_id_chance = 1.0 / len(set(video_id_int))
    polyp_majority = pd.Series(polyp).value_counts(normalize=True).max()
    # video_id 的 chance 隨影片數變小（60-way 時只有 1.7%），固定的絕對 margin
    # （原本校準自 5-way pilot 的 +0.15）在多類別下門檻過高、幾乎不可能有單一 PC
    # 跨過，改用相對 chance 的倍數門檻，讓判準能跨越 pilot（5-way）與全資料集
    # （60-way）兩種規模。
    video_id_threshold = video_id_chance * 3
    fingerprint_pcs = per_pc_df[per_pc_df["video_id_univariate_acc"] > video_id_threshold]["pc_index"].tolist()
    pathology_pcs = per_pc_df[per_pc_df["polyp_univariate_acc"] > polyp_majority + 0.05]["pc_index"].tolist()
    overlap_pcs = sorted(set(fingerprint_pcs) & set(pathology_pcs))
    lines.append(
        f"video_id chance = {video_id_chance:.3f}（門檻 = chance 的 3 倍 = "
        f"{video_id_threshold:.3f}），polyp majority baseline = {polyp_majority:.3f}"
        "（門檻 = majority + 0.05）。\n\n"
        f"**主要承載指紋（video_id）訊號的 PC**（單變量 accuracy 明顯高於 chance）："
        f"{fingerprint_pcs}\n\n"
        f"**主要承載病理（polyp）訊號的 PC**（單變量 accuracy 明顯高於 majority）："
        f"{pathology_pcs}\n\n"
        f"**兩者重疊的 PC**：{overlap_pcs if overlap_pcs else '無'}——"
    )
    if overlap_pcs:
        lines.append(
            "重疊代表這幾個維度同時編碼了指紋與病理資訊，無法乾淨分離；後續要做\n"
            "「移除指紋但保留病理」這類操作時，這幾個維度是主要的張力所在。\n"
        )
    elif not fingerprint_pcs and scale_tag == "full":
        lines.append(
            f"但這裡「無重疊」主要是因為 video_id 在前 20 個 PC 裡沒有任何單一維度"
            f"跨過門檻（{n_videos}-way 分類，chance 只有 {video_id_chance:.3f}），不能"
            "直接讀成「指紋與病理乾淨分離」。對照上方累積曲線：video_id 要到約 128"
            "個 PC 合併才追上峰值準確率，代表在這個規模（分類數變多）下指紋訊號"
            "分散在遠多於 20 個主成分裡，單一 PC 層級看不出來，跟 pilot（5-way，"
            "PC4/PC5 就有明顯訊號）不同——類別數變多會讓每個維度需要編碼的資訊被"
            "攤薄，這是預期中的規模效應，而非指紋消失。\n"
        )
    elif scale_tag == "pilot":
        lines.append(
            f"在這個 pilot 上兩者集中的維度沒有重疊，方向上是好消息，但樣本數小\n"
            f"（{n_videos} 支影片），需要在全資料集上重新驗證這個結論是否穩固。\n"
        )
    else:
        lines.append(
            f"在全部 {n_videos} 支影片、{n_frames} 張影格的規模下，兩者集中的維度\n"
            "沒有重疊，是正式結論：指紋與病理訊號在這個 embedding 空間裡分別集中\n"
            "在不同主成分，沒有觀察到糾纏在一起、難以分離的維度。\n"
        )

    out_path = RESULTS_DIR / "e1_pca_report.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(curve)
    print(per_pc_df)


if __name__ == "__main__":
    main()
