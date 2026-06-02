# Deteksi Penyakit Daun Kentang - Streamlit Cloud Fixed

Paket ini adalah versi perbaikan untuk deployment Streamlit Cloud.

## Perbaikan utama

1. Memperbaiki error install dependency pada Python 3.12 Streamlit Cloud.
2. Memakai TensorFlow modern yang tersedia untuk Python 3.12.
3. Memakai loader kompatibilitas untuk model `keras_model.h5` lama.
4. Memperbaiki tampilan agar nyaman dibaca pada mode light maupun dark.
5. `packages.txt` sengaja dikosongkan agar tidak diproses sebagai daftar apt package palsu.

## Cara deploy ke Streamlit Cloud

1. Extract ZIP ini.
2. Upload semua isi folder ke root GitHub repository.
3. Pastikan file `requirements.txt`, `packages.txt`, `main.py`, `model_loader.py`, `labels.txt`, dan `keras_model.h5` berada di root repo.
4. Di Streamlit Cloud, entry point gunakan:

```text
main.py
```

5. Klik **Manage app** → **Clear cache and reboot**.

## Isi requirements.txt

```text
streamlit>=1.36,<2
tensorflow==2.21.0
numpy>=1.26,<3
pillow>=10,<13
h5py>=3.12,<4
plotly>=5.24,<7
```

## Catatan

Prediksi dari model ini bersifat bantuan awal dan tidak menggantikan pemeriksaan ahli pertanian.
