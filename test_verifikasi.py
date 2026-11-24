from modules import data_loader, pearson_model, kmeans_model

mats, masalah = data_loader.muat()
assert not masalah, masalah
B, M25, M27 = [mats[k] for k in data_loader.SHEET]
gagal = 0


def cek(nama, nilai, harapan, tol=5e-4):
    global gagal
    ok = abs(nilai - harapan) <= tol if isinstance(harapan, float) else nilai == harapan
    gagal += (not ok)
    print(("OK    " if ok else "GAGAL ") + f"{nama}: program={nilai} | laporan={harapan}")


# ---- Pearson (Tabel 18, 19, 23, 34)
pb = pearson_model.hitung(B)
p = pb["pasangan"].set_index(["Produk A", "Produk B"])
cek("r id card-plakat (bulanan)", float(p.loc[("id card", "plakat"), "r"]), 0.9549)
cek("p id card-plakat", float(p.loc[("id card", "plakat"), "p-value"]), 0.0030)
cek("p_BH id card-plakat", float(p.loc[("id card", "plakat"), "p_BH"]), 0.2107)
cek("pasangan bulanan", pb["ringkasan"]["n_pasangan"], 153)
cek("positif bulanan", pb["ringkasan"]["positif"], 88)
cek("negatif bulanan", pb["ringkasan"]["negatif"], 65)
cek("p<0,05 bulanan", pb["ringkasan"]["p_di_bawah_005"], 8)
cek("signifikan BH bulanan", pb["ringkasan"]["signifikan_bh"], 0)
cek("median r bulanan", pb["ringkasan"]["median_r"], 0.1922)
p25 = pearson_model.hitung(M25)
q = p25["pasangan"].set_index(["Produk A", "Produk B"])
cek("r name tag-plakat (W02-W26)", float(q.loc[("name tag", "plakat"), "r"]), 0.5082)
cek("p<0,05 mingguan 25", p25["ringkasan"]["p_di_bawah_005"], 10)
cek("signifikan BH mingguan 25", p25["ringkasan"]["signifikan_bh"], 0)
p27 = pearson_model.hitung(M27)
cek("p<0,05 mingguan 27", p27["ringkasan"]["p_di_bawah_005"], 10)
cek("r min bulanan", pb["ringkasan"]["r_minimal_p005"], 0.811, 1e-3)

# ---- K-Means (Tabel 26, 28, 29, 31, 36)
for nama, m, k_lap, sil_lap in [
    ("bulanan", B, 7, 0.2952),
    ("mingguan utama", M25, 4, 0.0735),
    ("mingguan tambahan", M27, 6, 0.0673),
]:
    tabel, k_pilih, cad = kmeans_model.evaluasi_k(m)
    cek(f"K terpilih {nama}", k_pilih, k_lap)
    cek(f"aturan cadangan {nama}", cad, False)
    h = kmeans_model.kelompokkan(m, k_pilih)
    cek(f"Silhouette {nama}", h["silhouette"], sil_lap, 6e-4)
    if nama == "bulanan":
        cek("DBI bulanan", h["dbi"], 0.8415)
        cek("inertia bulanan", h["inertia"], 18.760, 2e-3)
        cek("ARI-LOO bulanan", h["ari_loo_rata"], 0.556, 2e-3)
        cek("ARI seed min bulanan", h["ari_seed_min"], 0.721, 2e-3)
        ukuran = sorted(h["anggota"].groupby("Cluster").size().tolist(), reverse=True)
        cek("ukuran cluster bulanan", ukuran, [4, 3, 3, 2, 2, 2, 2])
        c3 = h["anggota"].set_index("Produk").Cluster
        cek("sablon & nota satu cluster", bool(c3["sablon"] == c3["nota"]), True)

print("\nSEMUA PEMERIKSAAN LOLOS" if gagal == 0 else f"\n{gagal} PEMERIKSAAN GAGAL")
raise SystemExit(1 if gagal else 0)
