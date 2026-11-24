import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)

SEED, N_INIT, N_INIT_CEPAT = 42, 50, 10
KS = list(range(2, 9))
EPS = 1e-12


def zscore(x: np.ndarray) -> np.ndarray:
    """Z-score per baris (produk) dengan simpangan baku populasi (ddof=0)."""
    mu, sd = x.mean(axis=1, keepdims=True), x.std(axis=1, ddof=0, keepdims=True)
    z = np.zeros_like(x, dtype=float)
    ok = (sd > EPS).ravel()
    z[ok] = (x[ok] - mu[ok]) / sd[ok]
    return z


def _fit(X, k, seed=SEED, n_init=N_INIT):
    return KMeans(n_clusters=k, init="k-means++", n_init=n_init, random_state=seed).fit(X)


def kategori_silhouette(s: float) -> str:
    """Kategori kekuatan struktur (Tabel 25)."""
    if s <= 0.25:
        return "tidak ada struktur berarti"
    if s <= 0.50:
        return "lemah"
    if s <= 0.70:
        return "wajar"
    return "kuat"


def evaluasi_k(df: pd.DataFrame, ks=KS):
    """Evaluasi setiap K lalu pilih K terbaik. Mengembalikan (tabel, K terpilih, pakai_cadangan)."""
    x = df.values.astype(float)
    X = zscore(x)
    baris = []
    for k in ks:
        if k >= len(X):
            continue
        km = _fit(X, k)
        lab = km.labels_
        ukuran = np.bincount(lab, minlength=k)
        # ARI leave-one-period-out: buang satu periode, hitung ulang z-score + K-Means
        ari = [
            adjusted_rand_score(lab, _fit(zscore(np.delete(x, j, axis=1)), k, SEED, N_INIT_CEPAT).labels_)
            for j in range(x.shape[1])
        ]
        baris.append({
            "K": k,
            "Inertia": km.inertia_,
            "Silhouette": silhouette_score(X, lab),
            "DBI": davies_bouldin_score(X, lab),
            "Ukuran cluster": " / ".join(map(str, sorted(ukuran, reverse=True))),
            "Ukuran min": int(ukuran.min()),
            "ARI-LOO rata-rata": float(np.mean(ari)),
        })
    t = pd.DataFrame(baris)
    layak = t[t["Ukuran min"] >= 2].copy()
    cadangan = layak.empty
    if cadangan:
        layak = t.copy()
    layak["Skor peringkat"] = (
        layak["Silhouette"].rank(ascending=False)
        + layak["DBI"].rank(ascending=True)
        + layak["ARI-LOO rata-rata"].rank(ascending=False)
    ) / 3
    t = t.merge(layak[["K", "Skor peringkat"]], on="K", how="left")
    k_terpilih = int(layak.sort_values(["Skor peringkat", "K"]).iloc[0]["K"])
    return t, k_terpilih, cadangan


def kelompokkan(df: pd.DataFrame, k: int, ambang_frek_rendah: int = 16) -> dict:
    """Bentuk model K-Means pada K tertentu, lengkap dengan profil dan ukuran kestabilan."""
    x = df.values.astype(float)
    X = zscore(x)
    km = _fit(X, k)
    lab = km.labels_
    ss = silhouette_samples(X, lab)
    sil = float(silhouette_score(X, lab))
    ari_seed = [adjusted_rand_score(lab, _fit(X, k, s, N_INIT_CEPAT).labels_) for s in range(20)]
    ari_loo = [
        adjusted_rand_score(lab, _fit(zscore(np.delete(x, j, axis=1)), k, SEED, N_INIT_CEPAT).labels_)
        for j in range(x.shape[1])
    ]
    total = df.sum(axis=1)
    anggota = pd.DataFrame({
        "Produk": df.index,
        "Cluster": lab + 1,
        "Total frekuensi": total.values.astype(int),
        "Silhouette sampel": ss,
        "Frekuensi rendah": np.where(total.values <= ambang_frek_rendah, "Ya", ""),
    }).sort_values(["Cluster", "Produk"]).reset_index(drop=True)

    profil = []
    for c in range(k):
        idx = np.where(lab == c)[0]
        zc = X[idx].mean(axis=0)
        profil.append({
            "Cluster": f"C{c + 1}",
            "Anggota": ", ".join(df.index[idx]),
            "Jumlah": len(idx),
            "Total frekuensi": int(total.values[idx].sum()),
            "Silhouette rata-rata": float(ss[idx].mean()),
            "Periode puncak": df.columns[int(np.argmax(zc))],
            "Periode terendah": df.columns[int(np.argmin(zc))],
        })
    pca = PCA(n_components=2, random_state=SEED).fit(X)
    return {
        "K": k,
        "labels": lab,
        "X": X,
        "centroid": km.cluster_centers_,
        "inertia": float(km.inertia_),
        "silhouette": sil,
        "dbi": float(davies_bouldin_score(X, lab)),
        "kategori": kategori_silhouette(sil),
        "ari_seed_min": float(np.min(ari_seed)),
        "ari_loo_rata": float(np.mean(ari_loo)),
        "anggota": anggota,
        "profil": pd.DataFrame(profil),
        "pca": pca.transform(X),
        "pca_var": float(pca.explained_variance_ratio_.sum()),
    }


def uji_permutasi(df: pd.DataFrame, n_perm: int = 100, seed: int = SEED) -> dict:
    """Bandingkan Silhouette terbaik data asli dengan data yang urutan waktunya diacak per produk."""
    x = df.values.astype(float)

    def terbaik(mat, n_init):
        X, s_best = zscore(mat), -1.0
        for k in KS:
            if k >= len(X):
                continue
            lab = _fit(X, k, SEED, n_init).labels_
            if np.bincount(lab, minlength=k).min() >= 2:
                s_best = max(s_best, silhouette_score(X, lab))
        return s_best

    asli = terbaik(x, N_INIT)
    rng = np.random.default_rng(seed)
    acak = np.array([terbaik(np.array([rng.permutation(r) for r in x]), N_INIT_CEPAT) for _ in range(n_perm)])
    return {
        "asli": float(asli),
        "rata_acak": float(acak.mean()),
        "persentil95": float(np.percentile(acak, 95)),
        "p_empiris": float((np.sum(acak >= asli) + 1) / (n_perm + 1)),
        "n_perm": n_perm,
        "acak": acak,
    }


def kesimpulan_struktur(hasil: dict, perm=None):
    """Terapkan kriteria 'kuat dan stabil' (Bab V.2). Mengembalikan (lolos?, daftar pemeriksaan)."""
    cek = [
        (hasil["silhouette"] > 0.50, f"Silhouette {hasil['silhouette']:.3f} (syarat > 0,50)"),
        (hasil["ari_loo_rata"] >= 0.80, f"ARI-LOO rata-rata {hasil['ari_loo_rata']:.3f} (syarat >= 0,80)"),
        (hasil["ari_seed_min"] >= 0.80, f"ARI antar-seed minimum {hasil['ari_seed_min']:.3f} (syarat >= 0,80)"),
    ]
    if perm is not None:
        cek.append((perm["p_empiris"] < 0.05, f"p empiris permutasi {perm['p_empiris']:.4f} (syarat < 0,05)"))
    teks = [("LOLOS - " if ok else "TIDAK - ") + t for ok, t in cek]
    if perm is None:
        teks.append("BELUM DIUJI - uji permutasi belum dijalankan")
    return (all(ok for ok, _ in cek) and perm is not None), teks
