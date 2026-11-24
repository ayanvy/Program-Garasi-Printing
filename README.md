# Program Analisis Pola Frekuensi Produk - Garasi Printing

Program penerapan (Bab VI) dari model Korelasi Pearson dan K-Means pada matriks frekuensi
transaksi produk Garasi Printing (Januari - Juni 2026). Antarmuka berbasis web lokal (Streamlit).

## Cara menjalankan
1. Pasang Python 3.9 atau lebih baru.
2. Di folder ini jalankan:
   ```
   pip install -r requirements.txt
   streamlit run app.py
   ```
   (atau klik dua kali `jalankan.bat` di Windows, `bash jalankan.sh` di Linux/Mac)
3. Browser terbuka otomatis di http://localhost:8501

Program langsung memakai data bawaan (`data/matriks_frekuensi.xlsx`). Berkas lain dengan format sheet yang sama
dapat diunggah lewat panel kiri.

## Struktur
```
app.py                      antarmuka (5 halaman) - alur program
modules/data_loader.py      Modul 1: baca dan periksa matriks frekuensi
modules/pearson_model.py    Modul 2: korelasi Pearson, uji t, koreksi Benjamini-Hochberg
modules/kmeans_model.py     Modul 3: z-score, pemilihan K, K-Means, Silhouette/DBI/ARI, uji permutasi
data/matriks_frekuensi.xlsx data bawaan: 3 matriks (bulanan 18x6, mingguan W02-W26 18x25, mingguan W01-W27 18x27)
test_verifikasi.py          mencocokkan hasil program dengan angka pada laporan (Bab IV dan V)
```

## Format berkas unggahan
Berkas .xlsx dengan sheet `Matriks_Frekuensi_Bulanan`, `Matriks_Frekuensi_Mingguan_25`, dan/atau
`Matriks_Frekuensi_Mingguan_27`. Kolom pertama = nama produk, kolom berikutnya = frekuensi tiap periode
(angka >= 0, minimal 4 produk dan 4 periode). Sheet yang tidak ada dilewati.

## Halaman
1. **Data Frekuensi** - ukuran matriks, tabel frekuensi, heatmap.
2. **Hubungan Pola Transaksi** - r, p-value, p_BH, arah, kekuatan, status tiap pasangan; heatmap; unduh Excel.
3. **Pengelompokan Produk** - K otomatis (aturan peringkat Silhouette, DBI, ARI-LOO) atau pilih sendiri;
   anggota dan profil cluster, centroid, PCA, tabel evaluasi K, uji permutasi opsional; unduh Excel.
4. **Ringkasan** - kesimpulan otomatis dan ketentuan membaca hasil.
5. **Dashboard** - ringkasan visual dengan bahasa sederhana untuk pembaca awam: produk terlaris, tren ramai-sepi, kelompok produk dengan pola mirip, pasangan produk yang naik-turunnya sejalan atau berlawanan, dan penilaian seberapa kuat hasilnya. Memakai hasil perhitungan yang sama dengan halaman lain.

## Pengaturan model (sama dengan Bab IV)
- Pearson: frekuensi asli, uji t dua sisi alpha 0,05, koreksi BH per matriks.
- K-Means: z-score per produk (simpangan baku populasi), K = 2..8, k-means++, n_init = 50, random_state = 42.
- Struktur dinilai kuat dan stabil bila Silhouette > 0,50, ARI-LOO >= 0,80, ARI antar-seed minimum >= 0,80,
  dan p permutasi < 0,05.

## Verifikasi
```
python test_verifikasi.py
```
Memeriksa 29 angka (mis. r id card-plakat 0,9549; K = 7, 4, 6; Silhouette 0,2952, 0,0735, 0,0673).
Semua harus berstatus OK.

## Catatan pembacaan hasil
Hasil bersifat eksploratif: tidak ada korelasi yang signifikan setelah koreksi BH, dan cluster tidak lebih baik
daripada data acak. Hasil tidak menunjukkan sebab-akibat maupun pembelian produk secara bersamaan.
