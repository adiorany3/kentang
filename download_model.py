"""Download the original keras_model.h5 from the source repository.

Run:
    python download_model.py
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

MODEL_URL = "https://raw.githubusercontent.com/adiorany3/kentang/main/keras_model.h5"
MODEL_PATH = Path(__file__).resolve().parent / "keras_model.h5"


def main() -> None:
    print("Downloading keras_model.h5...")

    with urllib.request.urlopen(MODEL_URL, timeout=120) as response:
        model_bytes = response.read()

    if len(model_bytes) < 1024 * 1024:
        raise RuntimeError("Download gagal: ukuran file terlalu kecil.")

    MODEL_PATH.write_bytes(model_bytes)
    print(f"Done: {MODEL_PATH} ({MODEL_PATH.stat().st_size / 1024 / 1024:.2f} MB)")


if __name__ == "__main__":
    main()
