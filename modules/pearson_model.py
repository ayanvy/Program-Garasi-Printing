import numpy as np
import pandas as pd
from scipy import stats

ALPHA = 0.05


def kekuatan(r: float) -> str:
    """Kategori kekuatan korelasi (Tabel 17)."""
    a = abs(r)
    if a < 0.2:
        return "sangat lemah"
    if a < 0.4:
        return "lemah"
    if a < 0.6:
        return "sedang"
    if a < 0.8:
        return "kuat"
    return "sangat kuat"


def hitung(df: pd.DataFrame) -> dict:
    """Hitung r, p-value, dan p_BH untuk semua pasangan produk.

    Mengembalikan dict: matriks_r, pasangan (DataFrame), ringkasan (dict).
    """
    n_periode = df.shape[1]
    produk = list(df.index)
    baris = []
    for i in range(len(produk)):
        for j in range(i + 1, len(produk)):
            x, y = df.iloc[i].values, df.iloc[j].values
            if np.std(x) == 0 or np.std(y) == 0:  # deret konstan: r tidak terdefinisi
                continue
            r, p = stats.pearsonr(x, y)
            baris.append((produk[i], produk[j], r, p))
    pas = pd.DataFrame(baris, columns=["Produk A", "Produk B", "r", "p-value"])
    pas["p_BH"] = stats.false_discovery_control(pas["p-value"].values, method="bh")
    pas["Arah"] = np.where(pas["r"] >= 0, "Positif", "Negatif")
    pas["Kekuatan"] = pas["r"].apply(kekuatan)
    pas["Status"] = np.select(
        [pas["p_BH"] < ALPHA, pas["p-value"] < ALPHA],
        ["Signifikan (setelah BH)", "Eksploratif (p<0,05, belum lolos BH)"],
        default="Tidak signifikan",
    )
    pas = pas.sort_values("r", key=np.abs, ascending=False).reset_index(drop=True)

    # |r| minimal agar p < 0,05 pada n periode (menunjukkan batas daya uji)
    t_krit = stats.t.ppf(1 - ALPHA / 2, n_periode - 2)
    r_min = t_krit / np.sqrt(n_periode - 2 + t_krit ** 2)
    ringkasan = {
        "n_periode": n_periode,
        "n_pasangan": len(pas),
        "positif": int((pas["r"] > 0).sum()),
        "negatif": int((pas["r"] < 0).sum()),
        "r_maks": float(pas["r"].max()),
        "r_min": float(pas["r"].min()),
        "median_r": float(pas["r"].median()),
        "p_di_bawah_005": int((pas["p-value"] < ALPHA).sum()),
        "harapan_kebetulan": round(len(pas) * ALPHA, 2),
        "signifikan_bh": int((pas["p_BH"] < ALPHA).sum()),
        "r_minimal_p005": float(r_min),
    }
    return {"matriks_r": df.T.corr(method="pearson"), "pasangan": pas, "ringkasan": ringkasan}
