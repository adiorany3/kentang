# Software Pendeteksi Penyakit Daun Kentang - Fixed Full Package

Paket ini adalah versi yang sudah disesuaikan agar aplikasi dapat berjalan kembali saat membuka model Keras lama `keras_model.h5`.

## Status paket

File `keras_model.h5` sudah disertakan di dalam ZIP ini, jadi Anda tidak perlu mengunduh model secara manual.

## Masalah yang diperbaiki

Error utama:

```text
Layer "functional_4" expects 1 input(s), but it received 2 input tensors
```

Perbaikan utama:

- Menghapus penggunaan standalone `keras` dan memakai `tensorflow.keras`.
- Mengunci TensorFlow ke versi `2.15.1` agar lebih kompatibel dengan model HDF5/Keras lama.
- Menambahkan loader model fallback yang dapat membuat salinan `keras_model.compat.h5` tanpa mengubah model asli.
- Menambahkan parser `labels.txt` yang mendukung format satu baris dan multi-baris.
- Menyertakan `keras_model.h5` langsung di paket final.

## Isi paket

```text
main.py                  # Aplikasi Streamlit fixed
localRun.py              # Runner webcam lokal fixed
keras_model.h5           # Model Keras asli yang sudah disertakan
labels.txt               # Label model
download_model.py        # Opsional: downloader ulang model dari repo asli
requirements.txt         # Dependency yang sudah dipin
runtime.txt              # Runtime Python 3.11 untuk Streamlit Cloud
.streamlit/config.toml   # Konfigurasi Streamlit
start_windows.bat        # Shortcut jalan di Windows
start_linux_mac.sh       # Shortcut jalan di Linux/macOS
test.py                  # Tes kamera sederhana
MODEL_FILE_NOTE.txt      # Catatan model
```

## Cara menjalankan lokal

Disarankan memakai Python 3.10 atau 3.11. Jangan gunakan Python 3.12 untuk paket ini.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependency:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Jalankan aplikasi:

```bash
streamlit run main.py
```

## Cara cepat Windows

Double click:

```text
start_windows.bat
```

## Cara cepat Linux/macOS

```bash
chmod +x start_linux_mac.sh
./start_linux_mac.sh
```

## Catatan penting

- Jangan install package `keras` secara terpisah.
- Jangan pakai Python 3.12 untuk paket ini.
- File `keras_model.compat.h5` akan dibuat otomatis jika loader perlu fallback kompatibilitas.
- Jika deploy ke Streamlit Cloud, pastikan semua file di paket ini ikut di-push ke repository, termasuk `keras_model.h5`.

## Kredit

Developed by: Galuh Adi Insani.
