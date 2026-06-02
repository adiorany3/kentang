# Developed by Galuh Adi Insani
# Dimohon jangan hilangkan pada bagian ini untuk menghargai hasil kerja keras developer.
# Fixed package for Streamlit Cloud: modern TensorFlow install + legacy H5 manual weight loading.

from __future__ import annotations

import datetime
import os
import re
from pathlib import Path
from typing import List, Tuple, Union

# Keep TensorFlow output quieter on Streamlit Cloud.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from model_loader import build_compatible_model, preprocess_pil_image

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "keras_model.h5"
LABELS_PATH = BASE_DIR / "labels.txt"
CURRENT_YEAR = datetime.datetime.now().year

st.set_page_config(
    page_title="Deteksi Penyakit Daun Kentang",
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
            padding: 0.6rem;
            margin-bottom: 1rem;
            background: linear-gradient(90deg, rgba(219,234,254,0.4) 0%, rgba(191,219,254,0.4) 100%);
            border-radius: 12px;
        }
        .disease-card {
            padding: 1.25rem;
            border-radius: 12px;
            margin-bottom: 1rem;
            box-shadow: 0 4px 10px rgba(0,0,0,0.06);
            background: white;
            border-left: 5px solid #1E3A8A;
        }
        .disease-title {
            font-size: 1.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            color: #111827;
        }
        .confidence-display {
            text-align: center;
            font-size: 1.25rem;
            font-weight: 700;
            color: #1E3A8A;
            margin: 0.5rem 0;
        }
        .confidence-bar {
            height: 8px;
            background-color: #e5e7eb;
            border-radius: 999px;
            margin: 0.5rem 0;
        }
        .confidence-bar-fill {
            height: 100%;
            border-radius: 999px;
        }
        .camera-container {
            background-color: white;
            padding: 1rem;
            border-radius: 12px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.06);
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


DISEASE_INFO = {
    "earlyblight": {
        "title": "Early Blight",
        "icon": "🍂",
        "description": "Penyakit busuk daun awal yang umumnya ditandai bercak cokelat melingkar pada daun.",
        "recommendation": (
            "Pisahkan bagian tanaman yang terinfeksi, perbaiki sirkulasi udara, hindari penyiraman dari atas, "
            "dan konsultasikan penggunaan fungisida sesuai anjuran ahli pertanian setempat."
        ),
    },
    "healthy": {
        "title": "Healthy",
        "icon": "✅",
        "description": "Daun tampak sehat berdasarkan pola visual yang dikenali model.",
        "recommendation": (
            "Pertahankan pemupukan seimbang, penyiraman cukup, monitoring rutin, dan sanitasi lahan."
        ),
    },
    "lateblight": {
        "title": "Late Blight",
        "icon": "⚠️",
        "description": "Penyakit busuk daun akhir yang dapat menyebar cepat pada kondisi lembap.",
        "recommendation": (
            "Segera isolasi tanaman terindikasi, kurangi kelembapan berlebih, buang jaringan terinfeksi dengan aman, "
            "dan minta konfirmasi ahli pertanian."
        ),
    },
    "tryagain": {
        "title": "Try Again",
        "icon": "🔁",
        "description": "Gambar kurang sesuai, kurang jelas, atau tidak cukup menyerupai daun kentang untuk diprediksi.",
        "recommendation": "Ambil ulang foto dengan pencahayaan cukup, daun berada di tengah gambar, dan latar belakang tidak terlalu ramai.",
    },
}


with st.sidebar:
    st.markdown(
        "<h2 style='text-align: center; color: #1E3A8A;'>🥔 Deteksi Penyakit Daun Kentang</h2>",
        unsafe_allow_html=True,
    )

    with st.expander("ℹ️ Tentang Aplikasi", expanded=True):
        st.markdown(
            """
            Aplikasi ini memprediksi kondisi daun kentang menggunakan computer vision dan model machine learning.

            **Kelas yang didukung:**
            - Early Blight
            - Healthy
            - Late Blight
            - TryAgain
            """
        )

    with st.expander("📌 Cara Penggunaan", expanded=False):
        st.markdown(
            """
            1. Ambil foto daun kentang atau upload gambar.
            2. Pastikan pencahayaan cukup dan objek daun terlihat jelas.
            3. Tunggu hasil prediksi dan confidence score.

            Prediksi ini adalah diagnosis awal, bukan pengganti pemeriksaan ahli.
            """
        )

    with st.expander("🛠️ Info Perbaikan", expanded=False):
        st.markdown(
            """
            Paket ini memakai loader kompatibilitas khusus untuk model `.h5` lama.
            Arsitektur model dibangun ulang, lalu bobot dibaca langsung dari file HDF5.
            Cara ini menghindari error deserialisasi Keras seperti `expects 1 input(s), but it received 2 input tensors`.
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


def normalize_label(label: str) -> str:
    return re.sub(r"[^a-z0-9]", "", label.lower())


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


@st.cache_resource(show_spinner=False)
def load_model_resource():
    return build_compatible_model(MODEL_PATH)


def validate_image(image_file) -> Tuple[bool, Union[Image.Image, str]]:
    max_size = 10 * 1024 * 1024

    try:
        file_size = getattr(image_file, "size", None)
        if file_size is None and hasattr(image_file, "getbuffer"):
            file_size = len(image_file.getbuffer())

        if file_size is not None and file_size > max_size:
            return False, "Ukuran file terlalu besar. Maksimal 10 MB."

        image = Image.open(image_file).convert("RGB")
        return True, image
    except Exception as error:
        return False, f"Gagal memproses gambar: {error}"


def predict_image(model, image: Image.Image) -> Tuple[int, float, np.ndarray]:
    processed_image = preprocess_pil_image(image)
    prediction = model.predict(processed_image, verbose=0)
    prediction = np.asarray(prediction)

    index = int(np.argmax(prediction[0]))
    confidence_score = float(prediction[0][index]) * 100.0
    return index, confidence_score, prediction


def display_confidence(score: float) -> None:
    color = "#22c55e" if score >= 80 else "#eab308" if score >= 50 else "#ef4444"
    st.markdown(
        f"""
        <div class="confidence-display">Confidence Score: {score:.2f}%</div>
        <div class="confidence-bar">
            <div class="confidence-bar-fill" style="width: {min(max(score, 0), 100)}%; background-color: {color};"></div>
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
            title={"text": "Confidence Score"},
            number={"suffix": "%", "valueformat": ".2f"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#1E3A8A"},
                "steps": [
                    {"range": [0, 50], "color": "#fee2e2"},
                    {"range": [50, 80], "color": "#fef9c3"},
                    {"range": [80, 100], "color": "#dcfce7"},
                ],
            },
        )
    )
    fig.update_layout(height=260, margin={"l": 20, "r": 20, "t": 40, "b": 20})
    return fig


def render_prediction(label: str, confidence_score: float) -> None:
    info = DISEASE_INFO.get(normalize_label(label), DISEASE_INFO["tryagain"])
    st.markdown(
        f"""
        <div class="disease-card">
            <div class="disease-title">{info['icon']} {info['title']}</div>
            <p><strong>Deskripsi:</strong> {info['description']}</p>
            <p><strong>Rekomendasi:</strong> {info['recommendation']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    display_confidence(confidence_score)


st.markdown(
    "<h1 class='main-header'>🥔 Deteksi Penyakit Daun Kentang</h1>",
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([1, 1])

with left_col:
    st.markdown("<div class='camera-container'>", unsafe_allow_html=True)
    st.subheader("📷 Ambil atau Upload Gambar")

    camera_image = st.camera_input("Ambil foto daun kentang")

    with st.expander("📤 Upload gambar"):
        uploaded_file = st.file_uploader(
            "Pilih file gambar",
            type=["jpg", "jpeg", "png", "webp"],
        )

    image_file = camera_image or uploaded_file
    image = None

    if image_file is not None:
        valid, result = validate_image(image_file)
        if valid:
            image = result
            st.image(image, caption="Gambar yang akan dianalisis", use_container_width=True)
        else:
            st.error(result)

    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    st.subheader("🔍 Hasil Analisis")

    if image is None:
        st.info("Silakan ambil foto atau upload gambar daun kentang terlebih dahulu.")
    else:
        try:
            with st.spinner("Memuat model dan memproses prediksi..."):
                model = load_model_resource()
                class_names = load_class_names()
                index, confidence_score, _prediction = predict_image(model, image)

            if index >= len(class_names):
                st.error(
                    f"Index prediksi `{index}` melebihi jumlah label `{len(class_names)}`. Periksa file labels.txt."
                )
            else:
                label = class_names[index]
                render_prediction(label, confidence_score)
                st.plotly_chart(create_gauge_chart(confidence_score), use_container_width=True)

        except Exception as error:
            st.error("Aplikasi gagal memuat model atau melakukan prediksi.")
            st.code(str(error))
            st.info(
                "Pastikan `keras_model.h5` berada di folder yang sama dengan `main.py` dan dependency berhasil ter-install."
            )

st.markdown("---")
st.markdown(
    f"<div class='footer'>© {CURRENT_YEAR} Deteksi Penyakit Daun Kentang | Fixed compatibility package</div>",
    unsafe_allow_html=True,
)
