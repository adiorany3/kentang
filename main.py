# Developed by Galuh Adi Insani
# Dimohon jangan hilangkan pada bagian ini untuk menghargai hasil kerja keras developer.
# Fixed package: TensorFlow/Keras compatibility for legacy keras_model.h5.

from __future__ import annotations

import datetime
import os
import re
import shutil
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Tuple, Union

# Keep TensorFlow logs quieter. Do not import standalone `keras`.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import h5py
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "keras_model.h5"
PATCHED_MODEL_PATH = BASE_DIR / "keras_model.compat.h5"
LABELS_PATH = BASE_DIR / "labels.txt"
MODEL_URL = "https://raw.githubusercontent.com/adiorany3/kentang/main/keras_model.h5"
CURRENT_YEAR = datetime.datetime.now().year
TARGET_SIZE = (224, 224)


st.set_page_config(
    page_title="Deteksi Penyakit Kentang",
    page_icon="🥔",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
        .main-header {
            color: #1E3A8A;
            font-weight: 700;
            text-align: center;
            padding: 0.5rem;
            margin-bottom: 1rem;
            background: linear-gradient(90deg, rgba(219,234,254,0.3) 0%, rgba(191,219,254,0.3) 100%);
            border-radius: 10px;
        }
        .disease-card {
            padding: 1.25rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            background: white;
            border-left: 4px solid #1E3A8A;
        }
        .disease-title {
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
            color: #111827;
        }
        .confidence-display {
            text-align: center;
            font-size: 1.25rem;
            font-weight: 600;
            color: #1E3A8A;
            margin: 0.5rem 0;
        }
        .confidence-bar {
            height: 6px;
            background-color: #e5e7eb;
            border-radius: 3px;
            margin: 0.5rem 0;
        }
        .confidence-bar-fill {
            height: 100%;
            border-radius: 3px;
        }
        .camera-container {
            background-color: white;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        }
        .footer {
            text-align: center;
            color: #6b7280;
            font-size: 0.9rem;
        }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.markdown(
        "<h2 style='text-align: center; color: #1E3A8A;'>🥔 Deteksi Penyakit Daun Kentang</h2>",
        unsafe_allow_html=True,
    )

    with st.expander("ℹ️ Tentang Aplikasi", expanded=True):
        st.markdown(
            """
            ### Tentang Software
            Aplikasi ini memprediksi penyakit pada daun kentang menggunakan computer vision dan machine learning.

            **Penyakit yang dapat dideteksi:**
            - ✅ **Healthy** - Daun kentang sehat
            - 🍂 **Early Blight** - Penyakit busuk daun awal
            - ⚠️ **Late Blight** - Penyakit busuk daun akhir
            - ❌ **TryAgain** - Gambar tidak sesuai / perlu dicoba ulang
            """
        )

    with st.expander("📌 Cara Penggunaan", expanded=False):
        st.markdown(
            """
            1. Foto daun kentang memakai kamera atau unggah gambar.
            2. Pastikan pencahayaan cukup dan objek daun terlihat jelas.
            3. Sistem akan menampilkan prediksi dan confidence score.

            > Prediksi aplikasi hanya diagnosis awal. Konfirmasi akhir tetap perlu dilakukan oleh ahli pertanian.
            """
        )

    with st.expander("📚 Sumber Data", expanded=False):
        st.markdown(
            """
            Data dikembangkan dengan memanfaatkan database Kaggle yang diproses menggunakan machine learning.
            [Potato Leaf Disease Dataset (Kaggle)](https://www.kaggle.com/datasets/muhammadardiputra/potato-leaf-disease-dataset)
            """
        )

    st.markdown("---")
    st.markdown(
        "<div style='text-align: center;'><h4>Developer Contact</h4></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='text-align: center;'><a href='https://www.linkedin.com/in/galuh-adi-insani-1aa0a5105/' target='_blank'>Galuh Adi Insani</a></div>",
        unsafe_allow_html=True,
    )


def _download_model_if_missing() -> None:
    """Download the original model if it is not packaged locally.

    The binary model is intentionally not generated or replaced. This only fetches
    the original repository asset when it is absent.
    """
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 0:
        return

    try:
        with urllib.request.urlopen(MODEL_URL, timeout=60) as response:
            model_bytes = response.read()

        if len(model_bytes) < 1024 * 1024:
            raise RuntimeError(
                "File model yang terunduh terlalu kecil. Kemungkinan unduhan gagal atau URL berubah."
            )

        MODEL_PATH.write_bytes(model_bytes)
    except (urllib.error.URLError, TimeoutError, RuntimeError) as error:
        raise FileNotFoundError(
            "File `keras_model.h5` belum tersedia dan auto-download gagal. "
            "Unduh manual dari repository GitHub asli, lalu letakkan di folder yang sama dengan `main.py`. "
            f"Detail: {error}"
        ) from error


def _load_keras_model(path: Path):
    """Load a Keras H5 model while tolerating old/new Keras signatures."""
    try:
        return tf.keras.models.load_model(path, compile=False, safe_mode=False)
    except TypeError:
        return tf.keras.models.load_model(path, compile=False)


def _remove_groups_from_h5_copy(source_path: Path, target_path: Path) -> Path:
    """Create a compatible H5 copy without a legacy `groups` config entry.

    Some exported Keras/Teachable Machine H5 files contain fields that can break
    loading on newer TensorFlow/Keras combinations. The original model is never
    modified; a `.compat.h5` copy is regenerated when needed.
    """
    if target_path.exists() and target_path.stat().st_mtime >= source_path.stat().st_mtime:
        return target_path

    shutil.copy2(source_path, target_path)

    with h5py.File(target_path, mode="r+") as h5_file:
        model_config = h5_file.attrs.get("model_config")

        if isinstance(model_config, bytes):
            model_config = model_config.decode("utf-8")

        if isinstance(model_config, str):
            cleaned_config = model_config.replace('"groups": 1,', "")
            cleaned_config = cleaned_config.replace(', "groups": 1', "")

            if cleaned_config != model_config:
                h5_file.attrs["model_config"] = cleaned_config
                h5_file.flush()

    return target_path


@st.cache_resource(show_spinner=False)
def load_model_resource():
    _download_model_if_missing()

    load_errors: List[str] = []

    try:
        return _load_keras_model(MODEL_PATH)
    except Exception as error:  # noqa: BLE001 - user-facing diagnostics are needed here.
        load_errors.append(f"original: {error}")

    try:
        compatible_model_path = _remove_groups_from_h5_copy(MODEL_PATH, PATCHED_MODEL_PATH)
        return _load_keras_model(compatible_model_path)
    except Exception as error:  # noqa: BLE001
        load_errors.append(f"compat-copy: {error}")

    details = "\n\n".join(load_errors)
    raise RuntimeError(
        "Model gagal dimuat. Penyebab paling umum: file `.h5` lama dibuka dengan "
        "kombinasi Keras/TensorFlow terbaru yang tidak kompatibel. Gunakan Python 3.10/3.11 "
        "dan install dependency dari `requirements.txt` paket ini.\n\n"
        f"Detail error:\n{details}"
    )


def load_class_names(labels_path: Path = LABELS_PATH) -> List[str]:
    if not labels_path.exists():
        raise FileNotFoundError(f"File label tidak ditemukan: {labels_path.name}")

    raw_text = labels_path.read_text(encoding="utf-8").strip()
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

    if len(lines) > 1:
        names = []
        for line in lines:
            match = re.match(r"^\s*\d+\s+(.+?)\s*$", line)
            names.append(match.group(1).strip() if match else line.strip())
        return names

    matches = re.findall(r"(?:^|\s)\d+\s+(.+?)(?=\s+\d+\s+|$)", raw_text)
    if matches:
        return [item.strip() for item in matches]

    return lines


def normalize_label(label: str) -> str:
    return re.sub(r"[^a-z0-9]", "", label.lower())


def validate_image(image_file) -> Tuple[bool, Union[Image.Image, str]]:
    max_size = 5 * 1024 * 1024

    try:
        file_size = getattr(image_file, "size", None)
        if file_size is None and hasattr(image_file, "getbuffer"):
            file_size = len(image_file.getbuffer())

        if file_size is not None and file_size > max_size:
            return False, "Ukuran file terlalu besar. Maksimal 5MB."

        image = Image.open(image_file).convert("RGB")
        return True, image
    except Exception as error:  # noqa: BLE001
        return False, f"Error memproses gambar: {error}"


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
    image_array = np.asarray(image, dtype=np.float32)
    image_array = (image_array / 127.5) - 1.0
    return np.expand_dims(image_array, axis=0)


def predict_image(model, image: Image.Image) -> Tuple[int, float, np.ndarray]:
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image, verbose=0)

    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]

    prediction = np.asarray(prediction)
    index = int(np.argmax(prediction[0]))
    confidence_score = float(prediction[0][index])
    return index, confidence_score, prediction


def display_confidence(score: float) -> None:
    color = "#22c55e" if score > 90 else "#eab308" if score > 70 else "#ef4444"
    st.markdown(
        f"""
        <div class="confidence-display">Confidence Score: {score:.2f}%</div>
        <div class="confidence-bar">
            <div class="confidence-bar-fill" style="width: {score}%; background-color: {color};"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def create_gauge_chart(score: float):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Confidence Score", "font": {"size": 16, "color": "#1E3A8A"}},
            number={"font": {"size": 20, "color": "#1E3A8A"}, "suffix": "%", "valueformat": ".2f"},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#1E3A8A"},
                "bar": {"color": "#1E3A8A" if score > 90 else "#eab308" if score > 70 else "#ef4444"},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "#e5e7eb",
                "steps": [
                    {"range": [0, 50], "color": "#fee2e2"},
                    {"range": [50, 80], "color": "#fef9c3"},
                    {"range": [80, 100], "color": "#dcfce7"},
                ],
                "threshold": {"line": {"color": "#16a34a", "width": 4}, "thickness": 0.75, "value": 90},
            },
        )
    )
    fig.update_layout(
        height=150,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def render_prediction(label: str, confidence_score: float) -> None:
    normalized = normalize_label(label)
    confidence_percent = confidence_score * 100

    if normalized == "tryagain":
        st.markdown(
            f"""
            <div class="disease-card" style="border-left-color: #ef4444;">
                <div class="disease-title" style="color: #c2410c;">Error: Gambar Tidak Sesuai</div>
                <p>Model mengindikasikan gambar tidak dapat diproses dengan baik atau bukan gambar daun kentang yang valid.</p>
                <p><i>Label terdeteksi: {label} ({confidence_percent:.2f}%)</i></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    if confidence_score <= 0.7:
        st.warning(
            f"Keyakinan prediksi rendah. Prediksi sementara: {label} "
            f"(Keyakinan: {confidence_percent:.2f}%). Coba ambil gambar dengan pencahayaan dan fokus yang lebih baik."
        )
        return

    if normalized == "earlyblight":
        title = "🍂 Early Blight (Busuk Daun Awal)"
        description = (
            "Disebabkan oleh jamur Alternaria solani. Gejala umum berupa bercak coklat "
            "berbentuk cincin konsentris, biasanya dimulai dari daun yang lebih tua."
        )
    elif normalized == "lateblight":
        title = "⚠️ Late Blight (Busuk Daun Akhir)"
        description = (
            "Disebabkan oleh Phytophthora infestans. Gejala umum berupa bercak hijau gelap "
            "hingga hitam yang cepat menyebar dan dapat disertai tepi daun berair."
        )
    elif normalized == "healthy":
        title = "✅ Healthy - Daun Kentang Sehat"
        description = "Daun kentang terlihat sehat, berwarna hijau merata, tanpa bercak atau lesi yang jelas."
    else:
        title = "Kondisi Tidak Dikenal"
        description = "Label terdeteksi tidak dikenali oleh aplikasi. Periksa kembali isi labels.txt."

    st.markdown(
        f"""
        <div class="disease-card">
            <div class="disease-title">{title}</div>
            <p>{description}</p>
            <p><i>Label terdeteksi: {label}</i></p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.markdown(
        "<h1 class='main-header'>🥔 Sistem Deteksi Penyakit Daun Kentang</h1>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("<h3>📷 Ambil / Upload Gambar Daun Kentang</h3>", unsafe_allow_html=True)
        st.markdown("<div class='camera-container'>", unsafe_allow_html=True)
        camera_image = st.camera_input(
            label="Capture Image",
            key="First Camera",
            label_visibility="collapsed",
        )
        with st.expander("📤 Upload Gambar"):
            uploaded_file = st.file_uploader(
                "Pilih file gambar",
                type=["jpg", "jpeg", "png"],
            )
        image_file = camera_image or uploaded_file
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<h3>📊 Hasil Analisis</h3>", unsafe_allow_html=True)

        try:
            with st.spinner("Memuat model..."):
                model = load_model_resource()
            class_names = load_class_names()
        except Exception as error:  # noqa: BLE001
            st.error(str(error))
            st.info(
                "Pastikan `keras_model.h5` dan `labels.txt` berada di folder yang sama dengan `main.py`. "
                "Gunakan Python 3.10/3.11 dan install dependency dari `requirements.txt` paket ini."
            )
            return

        if image_file is None:
            st.info("Silakan ambil foto atau unggah gambar daun kentang terlebih dahulu.")
            return

        is_valid, result = validate_image(image_file)
        if not is_valid:
            st.error(result)
            return

        image = result
        st.image(image, caption="Gambar Daun Kentang", use_container_width=True)

        with st.spinner("Menganalisis gambar..."):
            try:
                index, confidence_score, _prediction = predict_image(model, image)
            except Exception as error:  # noqa: BLE001
                st.error(f"Error saat melakukan prediksi: {error}")
                return

        if index >= len(class_names):
            st.error(
                f"Output model menghasilkan index {index}, tetapi labels.txt hanya memiliki {len(class_names)} label. "
                "Periksa kembali file labels.txt agar jumlah label sama dengan output model."
            )
            return

        label = class_names[index]
        confidence_percent = confidence_score * 100
        render_prediction(label, confidence_score)
        display_confidence(confidence_percent)
        st.plotly_chart(create_gauge_chart(confidence_percent), use_container_width=True)

    st.markdown("---")
    st.markdown(
        f"""
        <div class="footer">
            © {CURRENT_YEAR} Developed by:
            <a href="https://www.linkedin.com/in/galuh-adi-insani-1aa0a5105/" target="_blank">Galuh Adi Insani</a>
            with ❤️<br>All rights reserved.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
