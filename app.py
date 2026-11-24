import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from modules import data_loader, kmeans_model, pearson_model

st.set_page_config(page_title="Analisis Pola Produk - Garasi Printing", layout="wide")
WARNA = plt.get_cmap("tab10").colors


# ------------------------------------------------------------------ cache (hindari hitung ulang)
@st.cache_data(show_spinner=False)
def _pearson(df):
    return pearson_model.hitung(df)


@st.cache_data(show_spinner="Mengevaluasi K = 2 sampai 8 ...")
def _evaluasi_k(df):
    return kmeans_model.evaluasi_k(df)


@st.cache_data(show_spinner=False)
def _kmeans(df, k):
    return kmeans_model.kelompokkan(df, k)


@st.cache_data(show_spinner="Menjalankan uji permutasi (bisa memakan waktu) ...")
def _permutasi(df, n):
    return kmeans_model.uji_permutasi(df, n)


def excel(**sheets) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        for nama, d in sheets.items():
            d.to_excel(w, sheet_name=nama[:31], index=not isinstance(d.index, pd.RangeIndex))
    return buf.getvalue()


# ------------------------------------------------------------------ sidebar: sumber data + pilihan
st.sidebar.title("Garasi Printing")
st.sidebar.caption("Analisis pola frekuensi transaksi produk")
unggah = st.sidebar.file_uploader(
    "Unggah matriks frekuensi (.xlsx)",
    type="xlsx",
    help="Sheet: Matriks_Frekuensi_Bulanan, Matriks_Frekuensi_Mingguan_25, "
         "Matriks_Frekuensi_Mingguan_27. Kosongkan untuk memakai data bawaan (Jan-Jun 2026).",
)
try:
    matriks, masalah = data_loader.muat(unggah)
except Exception as e:
    matriks, masalah = {}, [f"Berkas tidak dapat dibaca: {e}"]
for m in masalah:
    st.sidebar.error(m)
if not matriks:
    st.error("Data belum dapat dipakai. Periksa berkas pada panel kiri.")
    st.stop()

halaman = st.sidebar.radio(
    "Halaman",
    ["Data Frekuensi", "Hubungan Pola Transaksi", "Pengelompokan Produk", "Ringkasan", "Dashboard"],
)
periode = st.sidebar.selectbox("Periode analisis", list(matriks))
df = matriks[periode]

# ------------------------------------------------------------------ halaman 1: data
if halaman == "Data Frekuensi":
    st.title("Data Frekuensi Produk")
    st.write(
        f"Matriks **{df.shape[0]} produk x {df.shape[1]} periode**, total frekuensi **{int(df.values.sum()):,}** transaksi. "
        "Isi sel = jumlah transaksi berbeda yang memuat produk pada periode tersebut (bukan nominal rupiah)."
        .replace(",", ".")
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Produk", df.shape[0])
    c2.metric("Periode", df.shape[1])
    c3.metric("Sel bernilai nol", f"{(df.values == 0).mean() * 100:.1f}%")
    st.dataframe(df.astype(int))

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(np.log1p(df.values), aspect="auto", cmap="Blues")
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df.index)
    n = df.shape[1]
    langkah = 1 if n <= 12 else 2
    ax.set_xticks(range(0, n, langkah))
    ax.set_xticklabels(df.columns[::langkah], rotation=60, fontsize=8)
    fig.colorbar(im, label="log(1 + frekuensi)")
    ax.set_title("Heatmap frekuensi produk")
    st.pyplot(fig)

# ------------------------------------------------------------------ halaman 2: Pearson
elif halaman == "Hubungan Pola Transaksi":
    st.title("Korelasi Pearson antarproduk")
    st.caption("Mengukur kemiripan naik-turunnya frekuensi dua produk. "
               "Bukan bukti sebab-akibat dan bukan bukti dibeli bersamaan.")
    h = _pearson(df)
    rg = h["ringkasan"]
    pas = h["pasangan"]

    c = st.columns(4)
    c[0].metric("Pasangan diuji", rg["n_pasangan"])
    c[1].metric("p < 0,05 (sebelum koreksi)", rg["p_di_bawah_005"],
                help=f"Dengan kebetulan saja diharapkan sekitar {rg['harapan_kebetulan']}")
    c[2].metric("Signifikan setelah BH", rg["signifikan_bh"])
    c[3].metric("Median r", f"{rg['median_r']:.3f}")
    if rg["signifikan_bh"] == 0:
        st.warning(
            f"Tidak ada pasangan yang signifikan setelah koreksi BH. Pasangan dengan p < 0,05 hanya boleh dibaca sebagai "
            f"**temuan eksploratif**. Dengan {rg['n_periode']} periode, |r| minimal agar p < 0,05 adalah {rg['r_minimal_p005']:.3f}."
        )
    else:
        st.success(f"{rg['signifikan_bh']} pasangan signifikan setelah koreksi BH (q < 0,05).")

    fig, ax = plt.subplots(figsize=(8, 7))
    R = h["matriks_r"]
    im = ax.imshow(R.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(R)))
    ax.set_xticklabels(R.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(R)))
    ax.set_yticklabels(R.index, fontsize=8)
    fig.colorbar(im, label="r Pearson")
    ax.set_title("Heatmap korelasi Pearson")
    st.pyplot(fig)

    st.subheader("Pasangan produk")
    cari = st.text_input("Cari produk (opsional)", placeholder="contoh: plakat")
    status = st.multiselect("Status", sorted(pas["Status"].unique()), default=sorted(pas["Status"].unique()))
    tampil = pas[pas["Status"].isin(status)]
    if cari:
        tampil = tampil[tampil["Produk A"].str.contains(cari, case=False)
                        | tampil["Produk B"].str.contains(cari, case=False)]
    st.dataframe(tampil.style.format({"r": "{:.4f}", "p-value": "{:.4f}", "p_BH": "{:.4f}"}), height=380)
    st.download_button("Unduh hasil Pearson (.xlsx)", excel(Pasangan=pas, Matriks_r=h["matriks_r"]),
                       "hasil_pearson.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ------------------------------------------------------------------ halaman 3: K-Means
elif halaman == "Pengelompokan Produk":
    st.title("Pengelompokan Produk dengan K-Means")
    st.caption("Fitur = z-score per produk, sehingga yang dibandingkan adalah bentuk naik-turunnya frekuensi, "
               "bukan volumenya.")
    tabel, k_auto, cadangan = _evaluasi_k(df)

    mode = st.radio("Jumlah cluster (K)", ["Otomatis (aturan penelitian)", "Pilih sendiri"], horizontal=True)
    k = k_auto if mode.startswith("Otomatis") else st.slider("K", 2, min(8, len(df) - 1), k_auto)
    st.info(
        f"K terpilih oleh aturan peringkat (Silhouette, DBI, ARI-LOO): **{k_auto}**. K yang dipakai sekarang: **{k}**."
        + (" Aturan cadangan dipakai karena tidak ada K yang layak." if cadangan else "")
    )
    h = _kmeans(df, k)

    c = st.columns(4)
    c[0].metric("Silhouette", f"{h['silhouette']:.3f}", h["kategori"], delta_color="off")
    c[1].metric("Davies-Bouldin", f"{h['dbi']:.3f}")
    c[2].metric("ARI-LOO rata-rata", f"{h['ari_loo_rata']:.3f}")
    c[3].metric("ARI antar-seed (min)", f"{h['ari_seed_min']:.3f}")

    st.subheader("Cluster hasil pengelompokan")
    st.dataframe(h["profil"].style.format({"Silhouette rata-rata": "{:.3f}", "Total frekuensi": "{:,}".format}))
    if (h["anggota"]["Frekuensi rendah"] == "Ya").any():
        st.caption(
            "Produk berfrekuensi rendah (<= 16 transaksi): "
            + ", ".join(h["anggota"].loc[h["anggota"]["Frekuensi rendah"] == "Ya", "Produk"])
            + ". Z-score produk ini mudah melonjak akibat satu atau dua transaksi, jadi posisinya dibaca hati-hati."
        )

    g1, g2 = st.columns(2)
    with g1:
        fig, ax = plt.subplots(figsize=(6, 4.2))
        for ci in range(k):
            ax.plot(df.columns, h["centroid"][ci], marker="o" if df.shape[1] <= 12 else None,
                    color=WARNA[ci % 10], label=f"C{ci + 1}")
        ax.axhline(0, color="gray", lw=0.6)
        ax.set_ylabel("z-score centroid")
        ax.set_title("Pola centroid tiap cluster")
        if df.shape[1] > 12:
            ax.set_xticks(range(0, df.shape[1], 3))
            ax.tick_params(axis="x", rotation=60, labelsize=7)
        ax.legend(fontsize=7, ncol=2)
        st.pyplot(fig)
    with g2:
        fig, ax = plt.subplots(figsize=(6, 4.2))
        for ci in range(k):
            idx = np.where(h["labels"] == ci)[0]
            ax.scatter(h["pca"][idx, 0], h["pca"][idx, 1], color=WARNA[ci % 10], label=f"C{ci + 1}", s=50)
        for i, nm in enumerate(df.index):
            ax.annotate(nm, h["pca"][i], fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_title(f"Proyeksi PCA 2D (menjelaskan {h['pca_var'] * 100:.0f}% varians)")
        st.pyplot(fig)

    with st.expander("Tabel evaluasi K = 2 sampai 8"):
        st.dataframe(tabel.style.format({"Inertia": "{:.3f}", "Silhouette": "{:.3f}", "DBI": "{:.3f}",
                                         "ARI-LOO rata-rata": "{:.3f}", "Skor peringkat": "{:.3f}"}))
        st.caption("K tanpa skor peringkat dianggap tidak layak karena ada cluster beranggota satu produk.")

    st.subheader("Uji pembanding dengan data acak (opsional)")
    st.write("Urutan waktu tiap produk diacak, lalu Silhouette terbaik data asli dibandingkan dengan data acak.")
    n_perm = st.select_slider("Jumlah permutasi", [50, 100, 200], value=100)
    if st.button("Jalankan uji permutasi"):
        st.session_state[f"perm_{periode}"] = _permutasi(df, n_perm)
    perm = st.session_state.get(f"perm_{periode}")
    if perm:
        c = st.columns(4)
        c[0].metric("Silhouette terbaik (asli)", f"{perm['asli']:.3f}")
        c[1].metric("Rata-rata data acak", f"{perm['rata_acak']:.3f}")
        c[2].metric("Persentil 95 acak", f"{perm['persentil95']:.3f}")
        c[3].metric("p empiris", f"{perm['p_empiris']:.4f}")
    st.session_state[f"hasil_{periode}"] = (h, perm)

    st.download_button("Unduh hasil K-Means (.xlsx)",
                       excel(Anggota=h["anggota"], Profil=h["profil"], Evaluasi_K=tabel),
                       "hasil_kmeans.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ------------------------------------------------------------------ halaman 5: dashboard (bahasa awam)
elif halaman == "Dashboard":
    def angka(n):
        return f"{int(n):,}".replace(",", ".")

    def rapikan(ax):
        ax.spines[["top", "right"]].set_visible(False)

    total_produk = df.sum(axis=1).sort_values(ascending=False)
    total_periode = df.sum(axis=0)
    total_semua = float(total_produk.sum())
    p_puncak, p_sepi = total_periode.idxmax(), total_periode.idxmin()

    st.title("Dashboard Hasil Analisis")
    st.caption(f"Ringkasan sederhana dari analisis data transaksi Garasi Printing. Periode: {periode}")

    # --- angka utama
    c = st.columns(4)
    c[0].metric("Total transaksi", angka(total_semua))
    c[1].metric("Jenis produk", df.shape[0])
    c[2].metric("Produk paling laris", total_produk.index[0].title(),
                f"{total_produk.iloc[0] / total_semua * 100:.0f}% dari semua transaksi", delta_color="off")
    c[3].metric("Periode paling ramai", p_puncak, f"{angka(total_periode.max())} transaksi", delta_color="off")
    st.divider()

    # --- 1. produk terlaris
    st.subheader("1. Produk apa yang paling sering dipesan?")
    top = total_produk.head(10)[::-1]
    g1, g2 = st.columns([3, 2])
    with g1:
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        batang = ax.barh([n.title() for n in top.index], top.values, color=WARNA[0])
        ax.bar_label(batang, labels=[angka(v) for v in top.values], padding=3, fontsize=8)
        ax.set_xlabel("Jumlah transaksi")
        ax.set_xlim(0, top.max() * 1.15)
        rapikan(ax)
        st.pyplot(fig)
    with g2:
        dua = total_produk.head(2)
        st.markdown(
            f"- **{dua.index[0].title()}** dan **{dua.index[1].title()}** bersama-sama menyumbang "
            f"**{dua.sum() / total_semua * 100:.0f}%** dari seluruh transaksi.\n"
            f"- Dari {df.shape[0]} jenis produk, **{int((total_produk / total_semua < 0.02).sum())} produk** "
            f"hanya menyumbang kurang dari 2% transaksi per produknya.\n"
            f"- Satu transaksi dihitung satu kali per produk, jadi angka ini adalah jumlah pesanan, bukan rupiah."
        )
    st.divider()

    # --- 2. tren waktu
    st.subheader("2. Kapan transaksi ramai dan kapan sepi?")
    g1, g2 = st.columns([3, 2])
    with g1:
        fig, ax = plt.subplots(figsize=(6.5, 3.6))
        ax.plot(total_periode.index, total_periode.values, color=WARNA[0], marker="o" if df.shape[1] <= 12 else None)
        ax.scatter([p_puncak], [total_periode.max()], color="green", zorder=3, s=60, label=f"Paling ramai ({p_puncak})")
        ax.scatter([p_sepi], [total_periode.min()], color="red", zorder=3, s=60, label=f"Paling sepi ({p_sepi})")
        ax.set_ylabel("Jumlah transaksi")
        ax.set_ylim(0, total_periode.max() * 1.15)
        if df.shape[1] > 12:
            ax.set_xticks(range(0, df.shape[1], 3))
            ax.tick_params(axis="x", rotation=60, labelsize=8)
        ax.legend(fontsize=8)
        rapikan(ax)
        st.pyplot(fig)
    with g2:
        st.markdown(
            f"- Paling ramai pada **{p_puncak}** ({angka(total_periode.max())} transaksi).\n"
            f"- Paling sepi pada **{p_sepi}** ({angka(total_periode.min())} transaksi).\n"
            f"- Rata-rata **{total_periode.mean():.0f} transaksi** per periode."
        )
    st.divider()

    # --- 3. kelompok produk
    st.subheader("3. Produk mana yang pola ramainya mirip?")
    st.write("Produk dikelompokkan otomatis berdasarkan kemiripan **naik-turun** pesanannya dari waktu ke waktu, "
             "bukan berdasarkan seberapa banyak dipesan.")
    tabel_k, k_auto, _ = _evaluasi_k(df)
    hk = _kmeans(df, k_auto)
    profil = hk["profil"]
    kolom = st.columns(min(3, len(profil)))
    for i, baris in profil.iterrows():
        with kolom[i % len(kolom)]:
            with st.container(border=True):
                st.markdown(f"**Kelompok {i + 1}**")
                st.write(baris["Anggota"].title().replace(", ", " · "))
                if baris["Jumlah"] == 1:
                    st.caption(f"Berdiri sendiri. Paling ramai {baris['Periode puncak']}, paling sepi {baris['Periode terendah']}.")
                else:
                    st.caption(f"Cenderung ramai di {baris['Periode puncak']} dan sepi di {baris['Periode terendah']}.")
    rendah = hk["anggota"].loc[hk["anggota"]["Frekuensi rendah"] == "Ya", "Produk"].tolist()
    if rendah:
        st.caption("Produk yang jarang dipesan (" + ", ".join(rendah) + ") polanya mudah berubah hanya karena satu-dua pesanan, "
                   "jadi posisinya dalam kelompok jangan dipandang pasti.")
    st.divider()

    # --- 4. pasangan produk
    st.subheader("4. Produk mana yang naik-turunnya sejalan?")
    hp = _pearson(df)
    pas = hp["pasangan"]
    sejalan = pas[pas["r"] > 0].head(5)
    berlawan = pas[pas["r"] < 0].head(5)
    g1, g2 = st.columns(2)
    for kol, judul, data, ket in [
        (g1, "Cenderung ramai bersamaan", sejalan, "Saat yang satu ramai, yang lain cenderung ikut ramai."),
        (g2, "Cenderung berlawanan", berlawan, "Saat yang satu ramai, yang lain cenderung sepi."),
    ]:
        with kol:
            st.markdown(f"**{judul}**")
            st.caption(ket)
            if data.empty:
                st.write("Tidak ada pasangan seperti ini pada data.")
            for _, b in data.iterrows():
                st.write(f"{b['Produk A'].title()} dan {b['Produk B'].title()}")
                st.progress(float(min(abs(b["r"]), 1.0)), text=f"Kemiripan {abs(b['r']) * 100:.0f}% ({b['Kekuatan']})")
    st.divider()

    # --- 5. seberapa bisa dipercaya
    st.subheader("5. Seberapa kuat hasil ini?")
    rg = hp["ringkasan"]
    perm = st.session_state.get(f"hasil_{periode}", (None, None))[1]
    lolos, _ = kmeans_model.kesimpulan_struktur(hk, perm)
    if rg["signifikan_bh"] == 0 and not lolos:
        st.warning(
            "**Anggap sebagai gambaran awal, belum bukti pasti.** Pola kemiripan antarproduk yang terlihat bisa saja "
            f"terjadi karena kebetulan, sebab data yang dianalisis hanya {df.shape[1]} periode. "
            "Jangan dipakai sendirian sebagai dasar keputusan besar."
        )
    elif rg["signifikan_bh"] > 0 and lolos:
        st.success("Pola kemiripan antarproduk dan pengelompokan produk sama-sama tergolong kuat dan stabil.")
    else:
        st.info(
            "Sebagian hasil cukup meyakinkan, sebagian belum. Lihat halaman **Ringkasan** untuk rincian "
            "pemeriksaan tiap bagian."
        )
    st.markdown(
        "- Kemiripan pola **tidak berarti** satu produk menyebabkan produk lain ramai.\n"
        "- Kemiripan pola **tidak berarti** kedua produk dibeli dalam satu pesanan yang sama.\n"
        "- Untuk penjelasan teknis dan angka lengkapnya, buka halaman lain di panel kiri."
    )

# ------------------------------------------------------------------ halaman 4: ringkasan + ketentuan
else:
    st.title("Ringkasan dan Pembacaan Hasil")
    st.caption(f"Periode: {periode}")
    hp = _pearson(df)["ringkasan"]
    tabel, k_auto, _ = _evaluasi_k(df)
    hk = _kmeans(df, k_auto)
    perm = st.session_state.get(f"hasil_{periode}", (None, None))[1]
    lolos, cek = kmeans_model.kesimpulan_struktur(hk, perm)

    st.subheader("Korelasi Pearson")
    st.write(
        f"Dari {hp['n_pasangan']} pasangan produk, {hp['positif']} berkorelasi positif dan {hp['negatif']} negatif "
        f"(r antara {hp['r_min']:.3f} dan {hp['r_maks']:.3f}). Sebanyak {hp['p_di_bawah_005']} pasangan memiliki p < 0,05 "
        f"(kebetulan saja memberi sekitar {hp['harapan_kebetulan']}), dan **{hp['signifikan_bh']} signifikan setelah koreksi BH**."
    )
    st.subheader(f"K-Means (K = {k_auto}, aturan otomatis)")
    st.write(f"Silhouette {hk['silhouette']:.3f} ({hk['kategori']}).")
    for t in cek:
        st.write(("✅ " if t.startswith("LOLOS") else "⚠️ " if t.startswith("BELUM") else "❌ ")
                 + t.split(" - ", 1)[1])
    if lolos:
        st.success("Seluruh kriteria terpenuhi: struktur cluster tergolong kuat dan stabil.")
    else:
        st.warning("Struktur cluster belum terbukti kuat dan stabil. "
                   "Cluster dipakai sebagai gambaran deskriptif, bukan kelompok yang terbukti nyata.")

    st.subheader("Ketentuan pembacaan hasil")
    st.table(pd.DataFrame([
        ["Korelasi Pearson", "Signifikan hanya jika p_BH < 0,05. p < 0,05 saja disebut eksploratif."],
        ["Arah dan kekuatan r", "Menunjukkan kemiripan pola naik-turun, bukan sebab-akibat dan bukan pembelian bersamaan."],
        ["Cluster K-Means", "Kuat dan stabil bila Silhouette > 0,50, ARI-LOO >= 0,80, ARI antar-seed >= 0,80, dan p permutasi < 0,05."],
        ["Pearson dan K-Means", "Bukan dua bukti independen: jarak z-score sama dengan 2n(1 - r)."],
        ["Produk frekuensi rendah", "Z-score mudah melonjak oleh satu atau dua transaksi; baca hati-hati."],
        ["Data enam bulan", "Daya uji rendah; hasil bulanan dan mingguan tidak selalu searah."],
    ], columns=["Hasil", "Cara membaca"]).set_index("Hasil"))
