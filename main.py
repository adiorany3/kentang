# Developed by Galuh Adi Insani
# Dimohon jangan hilangkan pada bagian ini untuk menghargai hasil kerja keras developer.
# Fixed package for Streamlit Cloud: modern TensorFlow install + legacy H5 manual weight loading.
# UI/theme revision: readable in Streamlit light and dark modes.

from __future__ import annotations

import datetime
import html
import os
import re
from pathlib import Path
from typing import List, Tuple, Union

# Keep TensorFlow output quieter on Streamlit Cloud.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
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


def inject_theme_css() -> None:
    """Apply CSS that follows Streamlit's active theme variables.

    The previous UI used hard-coded white cards and dark text. That looked fine in
    light mode, but became hard to read in dark mode. This stylesheet avoids
    fixed black/white backgrounds and uses Streamlit theme variables instead.
    """
    st.markdown(
        """
        <style>
            :root {
                --kentang-radius-lg: 1.15rem;
                --kentang-radius-md: 0.85rem;
                --kentang-border: rgba(128, 128, 128, 0.28);
                --kentang-shadow: 0 0.75rem 2.25rem rgba(0, 0, 0, 0.10);
                --kentang-shadow-soft: 0 0.35rem 1.2rem rgba(0, 0, 0, 0.08);
                --kentang-success: #16a34a;
                --kentang-warning: #ca8a04;
                --kentang-danger: #dc2626;
                --kentang-info: #2563eb;
            }

            /* Streamlit global readability */
            .stApp,
            [data-testid="stAppViewContainer"],
            [data-testid="stSidebar"],
            [data-testid="stHeader"],
            [data-testid="stMarkdownContainer"] {
                color: var(--text-color) !important;
            }

            .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
                max-width: 1180px;
            }

            p, li, label, span, div {
                line-height: 1.62;
            }

            a {
                color: var(--primary-color) !important;
                font-weight: 650;
                text-decoration-thickness: 0.08em;
                text-underline-offset: 0.18em;
            }

            /* Hero */
            .kentang-hero {
                border: 1px solid var(--kentang-border);
                border-radius: var(--kentang-radius-lg);
                padding: clamp(1.25rem, 3vw, 2.1rem);
                margin-bottom: 1.35rem;
                background:
                    radial-gradient(circle at top left, rgba(34, 197, 94, 0.16), transparent 34%),
                    radial-gradient(circle at bottom right, rgba(37, 99, 235, 0.13), transparent 30%),
                    var(--secondary-background-color);
                box-shadow: var(--kentang-shadow-soft);
            }

            .kentang-hero h1 {
                color: var(--text-color) !important;
                font-size: clamp(2rem, 5vw, 3.4rem);
                line-height: 1.12;
                margin: 0 0 0.55rem 0;
                letter-spacing: -0.04em;
            }

            .kentang-hero p {
                max-width: 820px;
                margin: 0;
                color: color-mix(in srgb, var(--text-color) 78%, transparent);
                font-size: 1.04rem;
            }

            .kentang-badges {
                display: flex;
                flex-wrap: wrap;
                gap: 0.55rem;
                margin-top: 1rem;
            }

            .kentang-badge {
                display: inline-flex;
                align-items: center;
                gap: 0.4rem;
                padding: 0.35rem 0.68rem;
                border-radius: 999px;
                border: 1px solid var(--kentang-border);
                background: color-mix(in srgb, var(--secondary-background-color) 86%, var(--background-color));
                color: var(--text-color);
                font-size: 0.88rem;
                font-weight: 650;
            }

            /* Cards */
            .kentang-card,
            .disease-card,
            .instruction-card,
            .metric-card {
                border: 1px solid var(--kentang-border);
                border-radius: var(--kentang-radius-lg);
                background: var(--secondary-background-color);
                color: var(--text-color);
                box-shadow: var(--kentang-shadow-soft);
            }

            .kentang-card {
                padding: 1rem 1.05rem;
                margin-bottom: 1rem;
            }

            .disease-card {
                padding: clamp(1rem, 2.6vw, 1.45rem);
                margin-bottom: 1rem;
                border-left: 0.42rem solid var(--primary-color);
            }

            .disease-title {
                display: flex;
                align-items: center;
                gap: 0.55rem;
                font-size: clamp(1.35rem, 3vw, 1.85rem);
                font-weight: 800;
                line-height: 1.2;
                margin-bottom: 0.75rem;
                color: var(--text-color) !important;
            }

            .disease-card p {
                margin: 0.62rem 0;
                color: var(--text-color) !important;
            }

            .disease-card strong,
            .section-kicker {
                color: var(--text-color) !important;
                font-weight: 800;
            }

            .section-title {
                display: flex;
                align-items: center;
                gap: 0.5rem;
                margin: 0.25rem 0 0.9rem;
                color: var(--text-color) !important;
                font-size: clamp(1.25rem, 2.4vw, 1.65rem);
                font-weight: 800;
                letter-spacing: -0.02em;
            }

            .section-note {
                margin-top: -0.25rem;
                margin-bottom: 1rem;
                color: color-mix(in srgb, var(--text-color) 76%, transparent) !important;
                font-size: 0.96rem;
            }

            /* Confidence */
            .confidence-panel {
                padding: 1rem;
                border-radius: var(--kentang-radius-md);
                border: 1px solid var(--kentang-border);
                background: color-mix(in srgb, var(--secondary-background-color) 84%, var(--background-color));
                margin: 1rem 0;
            }

            .confidence-display {
                display: flex;
                align-items: baseline;
                justify-content: space-between;
                gap: 1rem;
                margin-bottom: 0.75rem;
                color: var(--text-color) !important;
            }

            .confidence-label {
                font-size: 0.95rem;
                font-weight: 750;
                color: color-mix(in srgb, var(--text-color) 75%, transparent);
            }

            .confidence-number {
                font-size: clamp(1.55rem, 5vw, 2.65rem);
                line-height: 1;
                font-weight: 900;
                letter-spacing: -0.04em;
            }

            .confidence-bar {
                width: 100%;
                height: 0.78rem;
                background: color-mix(in srgb, var(--text-color) 14%, transparent);
                border-radius: 999px;
                overflow: hidden;
                border: 1px solid var(--kentang-border);
            }

            .confidence-bar-fill {
                height: 100%;
                border-radius: 999px;
            }

            .confidence-help {
                margin-top: 0.75rem;
                font-size: 0.92rem;
                color: color-mix(in srgb, var(--text-color) 72%, transparent);
            }

            /* Streamlit widgets */
            [data-testid="stFileUploader"],
            [data-testid="stCameraInput"] {
                border: 1px dashed var(--kentang-border);
                border-radius: var(--kentang-radius-md);
                padding: 0.65rem;
                background: color-mix(in srgb, var(--secondary-background-color) 76%, var(--background-color));
            }

            [data-testid="stFileUploader"] section {
                background: transparent !important;
                color: var(--text-color) !important;
            }

            [data-testid="stFileUploader"] small,
            [data-testid="stFileUploader"] span,
            [data-testid="stFileUploader"] p,
            [data-testid="stCameraInput"] small,
            [data-testid="stCameraInput"] span,
            [data-testid="stCameraInput"] p {
                color: var(--text-color) !important;
            }

            .stButton > button,
            [data-testid="baseButton-secondary"],
            [data-testid="baseButton-primary"] {
                border-radius: 999px !important;
                font-weight: 800 !important;
                border: 1px solid var(--kentang-border) !important;
            }

            div[data-testid="stExpander"] {
                border-color: var(--kentang-border) !important;
                background: color-mix(in srgb, var(--secondary-background-color) 80%, var(--background-color)) !important;
                border-radius: var(--kentang-radius-md) !important;
            }

            div[data-testid="stExpander"] * {
                color: var(--text-color) !important;
            }

            [data-testid="stSidebar"] {
                border-right: 1px solid var(--kentang-border);
                background: var(--secondary-background-color);
            }

            [data-testid="stSidebar"] h1,
            [data-testid="stSidebar"] h2,
            [data-testid="stSidebar"] h3,
            [data-testid="stSidebar"] h4,
            [data-testid="stSidebar"] p,
            [data-testid="stSidebar"] li,
            [data-testid="stSidebar"] span {
                color: var(--text-color) !important;
            }

            .sidebar-title {
                text-align: center;
                color: var(--text-color) !important;
                font-size: 1.35rem;
                line-height: 1.25;
                margin: 0.25rem 0 1rem;
                font-weight: 900;
            }

            .sidebar-credit {
                text-align: center;
                padding: 0.8rem 0.5rem;
                border-radius: var(--kentang-radius-md);
                border: 1px solid var(--kentang-border);
                background: color-mix(in srgb, var(--secondary-background-color) 80%, var(--background-color));
            }

            .footer {
                text-align: center;
                color: color-mix(in srgb, var(--text-color) 68%, transparent) !important;
                font-size: 0.9rem;
                padding: 1.1rem 0 0.4rem;
            }

            /* Better responsive spacing on mobile */
            @media (max-width: 768px) {
                .block-container {
                    padding-left: 1rem;
                    padding-right: 1rem;
                    padding-top: 1rem;
                }

                .confidence-display {
                    align-items: flex-start;
                    flex-direction: column;
                    gap: 0.35rem;
                }
            }

            /* Fallback when CSS color-mix is not supported */
            @supports not (color: color-mix(in srgb, white 50%, black)) {
                .kentang-hero,
                .kentang-badge,
                .confidence-panel,
                [data-testid="stFileUploader"],
                [data-testid="stCameraInput"],
                div[data-testid="stExpander"],
                .sidebar-credit {
                    background: var(--secondary-background-color) !important;
                }

                .kentang-hero p,
                .section-note,
                .confidence-label,
                .confidence-help,
                .footer {
                    color: var(--text-color) !important;
                    opacity: 0.78;
                }
            }

            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_theme_css()


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
        "<div class='sidebar-title'>🥔 Deteksi Penyakit<br/>Daun Kentang</div>",
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
            - Try Again
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
        """
        <div class="sidebar-credit">
            <strong>Developer Contact</strong><br/>
            <a href="https://www.linkedin.com/in/galuh-adi-insani-1aa0a5105/" target="_blank">Galuh Adi Insani</a>
        </div>
        """,
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


def confidence_color(score: float) -> str:
    if score >= 80:
        return "#16a34a"
    if score >= 50:
        return "#ca8a04"
    return "#dc2626"


def confidence_message(score: float) -> str:
    if score >= 80:
        return "Confidence tinggi. Hasil model relatif kuat, tetapi tetap gunakan pemeriksaan visual dan konteks lapangan."
    if score >= 50:
        return "Confidence sedang. Disarankan mengambil foto lain dari sudut berbeda untuk pembanding."
    return "Confidence rendah. Ambil ulang foto dengan pencahayaan lebih baik dan latar belakang lebih bersih."


def display_confidence(score: float) -> None:
    safe_score = min(max(score, 0.0), 100.0)
    color = confidence_color(safe_score)
    message = html.escape(confidence_message(safe_score))

    st.markdown(
        f"""
        <div class="confidence-panel">
            <div class="confidence-display">
                <div class="confidence-label">Confidence Score</div>
                <div class="confidence-number" style="color: {color};">{safe_score:.2f}%</div>
            </div>
            <div class="confidence-bar" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="{safe_score:.2f}">
                <div class="confidence-bar-fill" style="width: {safe_score:.2f}%; background: {color};"></div>
            </div>
            <div class="confidence-help">{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prediction(label: str, confidence_score: float) -> None:
    info = DISEASE_INFO.get(normalize_label(label), DISEASE_INFO["tryagain"])

    title = html.escape(info["title"])
    icon = html.escape(info["icon"])
    description = html.escape(info["description"])
    recommendation = html.escape(info["recommendation"])

    st.markdown(
        f"""
        <div class="disease-card">
            <div class="disease-title"><span>{icon}</span><span>{title}</span></div>
            <p><strong>Deskripsi:</strong> {description}</p>
            <p><strong>Rekomendasi:</strong> {recommendation}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    display_confidence(confidence_score)


st.markdown(
    """
    <section class="kentang-hero">
        <h1>🥔 Deteksi Penyakit Daun Kentang</h1>
        <p>
            Upload atau ambil foto daun kentang, lalu aplikasi akan menampilkan hasil prediksi dan tingkat keyakinan model.
            Tampilan ini sudah dibuat adaptif agar teks dan kartu tetap terbaca pada mode terang maupun gelap.
        </p>
        <div class="kentang-badges">
            <span class="kentang-badge">🌗 Light/Dark friendly</span>
            <span class="kentang-badge">📷 Camera & upload</span>
            <span class="kentang-badge">🧠 Keras H5 compatible</span>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    with st.container(border=True):
        st.markdown("<div class='section-title'>📷 Ambil atau Upload Gambar</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-note'>Gunakan foto daun kentang yang jelas, tidak blur, dan pencahayaannya cukup.</div>",
            unsafe_allow_html=True,
        )

        camera_image = st.camera_input("Ambil foto daun kentang")

        with st.expander("📤 Upload gambar dari perangkat", expanded=True):
            uploaded_file = st.file_uploader(
                "Pilih file gambar",
                type=["jpg", "jpeg", "png", "webp"],
                help="Format yang didukung: JPG, JPEG, PNG, atau WEBP. Maksimal 10 MB.",
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

with right_col:
    with st.container(border=True):
        st.markdown("<div class='section-title'>🔍 Hasil Analisis</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-note'>Hasil akan muncul setelah model selesai membaca gambar.</div>",
            unsafe_allow_html=True,
        )

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

            except Exception as error:
                st.error("Aplikasi gagal memuat model atau melakukan prediksi.")
                st.code(str(error))
                st.info(
                    "Pastikan `keras_model.h5` berada di folder yang sama dengan `main.py` dan dependency berhasil ter-install."
                )

st.markdown("---")
st.markdown(
    f"<div class='footer'>© {CURRENT_YEAR} Deteksi Penyakit Daun Kentang | Streamlit Cloud compatibility + responsive theme package</div>",
    unsafe_allow_html=True,
)
