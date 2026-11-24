from pathlib import Path
import pandas as pd

# label yang ditampilkan di program -> nama sheet pada berkas Excel
SHEET = {
    "Bulanan (Jan-Jun, 6 periode)": "Matriks_Frekuensi_Bulanan",
    "Mingguan utama (W02-W26, 25 periode)": "Matriks_Frekuensi_Mingguan_25",
    "Mingguan tambahan (W01-W27, 27 periode)": "Matriks_Frekuensi_Mingguan_27",
}
DATA_BAWAAN = Path(__file__).resolve().parent.parent / "data" / "matriks_frekuensi.xlsx"


def periksa(df: pd.DataFrame, nama: str) -> list:
    """Kembalikan daftar masalah pada satu matriks (kosong = data layak)."""
    masalah = []
    if df.shape[0] < 4:
        masalah.append(f"{nama}: minimal 4 produk diperlukan, ditemukan {df.shape[0]}.")
    if df.shape[1] < 4:
        masalah.append(f"{nama}: minimal 4 periode diperlukan, ditemukan {df.shape[1]}.")
    if df.isna().any().any():
        masalah.append(f"{nama}: terdapat sel kosong atau bukan angka.")
    elif (df < 0).any().any():
        masalah.append(f"{nama}: terdapat frekuensi negatif.")
    if df.index.duplicated().any():
        masalah.append(f"{nama}: ada nama produk ganda.")
    return masalah


def muat(sumber=None):
    """Baca matriks dari berkas Excel (path atau objek unggahan).

    Mengembalikan (dict matriks yang layak, daftar masalah).
    Sheet yang tidak ada dilewati; nama sheet yang dikenali ada pada SHEET.
    """
    sumber = sumber if sumber is not None else DATA_BAWAAN
    xls = pd.ExcelFile(sumber)
    hasil, masalah = {}, []
    for label, sheet in SHEET.items():
        if sheet not in xls.sheet_names:
            continue
        df = pd.read_excel(xls, sheet_name=sheet, index_col=0)
        df.index = df.index.astype(str).str.strip()
        df = df.apply(pd.to_numeric, errors="coerce")
        m = periksa(df, label)
        masalah += m
        if not m:
            hasil[label] = df.astype(float)
    if not hasil and not masalah:
        masalah.append("Tidak ada sheet matriks yang dikenali. Nama sheet yang dibutuhkan: "
                       + ", ".join(SHEET.values()))
    return hasil, masalah
