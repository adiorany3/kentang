# Deteksi Penyakit Daun Kentang - Streamlit Cloud Fixed

Paket ini adalah versi final untuk deployment Streamlit Cloud.

## Penyebab error dari log

Streamlit Cloud memakai Python 3.12.13, tetapi `requirements.txt` di repo masih meminta:

```txt
tensorflow==2.15.1
```

TensorFlow 2.15.1 tidak punya wheel yang cocok untuk Python 3.12, sehingga dependency gagal di-install.

## File yang wajib ada di root repo

- `main.py`
- `model_loader.py`
- `keras_model.h5`
- `labels.txt`
- `requirements.txt`
- `runtime.txt`
- `.streamlit/config.toml`

## Cara deploy ulang di Streamlit Cloud

1. Extract ZIP ini.
2. Upload/replace semua file ke root repo GitHub `adiorany3/kentang` branch `main`.
3. Pastikan file lama `requirements.txt` benar-benar terganti.
4. Isi `requirements.txt` harus memuat `tensorflow==2.21.0`, bukan `tensorflow==2.15.1`.
5. Commit dan push.
6. Di Streamlit Cloud, klik `Manage app` -> `Reboot app` atau `Clear cache and reboot`.

## Entry point

```txt
main.py
```

## Menjalankan lokal

Direkomendasikan Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run main.py
```

Windows:

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run main.py
```

## Catatan teknis

Model `.h5` lama tidak di-load dengan `keras.models.load_model()` secara langsung, karena pada Keras modern dapat memunculkan error nested input:

```txt
Layer "functional_4" expects 1 input(s), but it received 2 input tensors
```

File `model_loader.py` membangun ulang arsitektur kompatibel dan membaca bobot langsung dari `keras_model.h5`.

## Catatan Error `Unable to locate package #`

Jika Streamlit Cloud menampilkan error seperti:

```txt
E: Unable to locate package #
E: Unable to locate package No
E: Unable to locate package extra
```

penyebabnya adalah `packages.txt` berisi komentar. Streamlit Cloud mengirim isi `packages.txt` ke `apt-get`, sehingga komentar ikut dianggap sebagai nama paket.

Solusi: kosongkan `packages.txt` atau hapus file tersebut dari repository. Paket ini sudah menyertakan `packages.txt` kosong agar file lama yang bermasalah bisa tertimpa saat di-upload ulang.
