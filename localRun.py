# Developed by Galuh Adi Insani
# Local webcam runner with TensorFlow/Keras compatibility fix.

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import h5py
import numpy as np
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "keras_model.h5"
PATCHED_MODEL_PATH = BASE_DIR / "keras_model.compat.h5"
LABELS_PATH = BASE_DIR / "labels.txt"
TARGET_SIZE = (224, 224)


def load_class_names(labels_path: Path):
    raw_text = labels_path.read_text(encoding="utf-8").strip()
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

    if len(lines) > 1:
        names = []
        for line in lines:
            match = re.match(r"^\s*\d+\s+(.+?)\s*$", line)
            names.append(match.group(1).strip() if match else line.strip())
        return names

    matches = re.findall(r"(?:^|\s)\d+\s+(.+?)(?=\s+\d+\s+|$)", raw_text)
    return [item.strip() for item in matches] if matches else lines


def load_keras_model(path: Path):
    try:
        return tf.keras.models.load_model(path, compile=False, safe_mode=False)
    except TypeError:
        return tf.keras.models.load_model(path, compile=False)


def make_compat_model_copy(source_path: Path, target_path: Path) -> Path:
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


def load_model_resource():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("File keras_model.h5 tidak ditemukan. Jalankan download_model.py terlebih dahulu.")

    try:
        return load_keras_model(MODEL_PATH)
    except Exception as original_error:  # noqa: BLE001
        print(f"Gagal load model original, mencoba compat copy: {original_error}")

    compat_path = make_compat_model_copy(MODEL_PATH, PATCHED_MODEL_PATH)
    return load_keras_model(compat_path)


def preprocess(image):
    image = cv2.resize(image, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.asarray(image, dtype=np.float32)
    image = (image / 127.5) - 1.0
    return np.expand_dims(image, axis=0)


def main():
    np.set_printoptions(suppress=True)

    model = load_model_resource()
    class_names = load_class_names(LABELS_PATH)

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Kamera tidak dapat dibuka. Coba ganti index kamera menjadi 1.")

    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                print("Frame kamera tidak terbaca.")
                break

            cv2.imshow("Webcam Image", cv2.resize(frame, TARGET_SIZE))

            prediction = model.predict(preprocess(frame), verbose=0)
            if isinstance(prediction, (list, tuple)):
                prediction = prediction[0]

            index = int(np.argmax(prediction[0]))
            confidence_score = float(prediction[0][index])
            class_name = class_names[index] if index < len(class_names) else f"Unknown index {index}"

            print(f"Class: {class_name} | Confidence Score: {confidence_score * 100:.2f}%")

            keyboard_input = cv2.waitKey(1)
            if keyboard_input == 27:
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
