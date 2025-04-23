# Software Pendeteksi Penyakit Daun Kentang

## Deskripsi
Proyek ini menggunakan machine learning dan computer vision untuk mendeteksi penyakit pada daun kentang. Saat ini, sistem dapat mendeteksi kondisi berikut:
- Healthy (Daun Kentang Sehat)
- Early Blight (Busuk Daun Awal)
- Late Blight (Busuk Daun Akhir)

Sistem memberikan prediksi dengan tingkat kepercayaan (confidence score) untuk membantu diagnosis awal.

## Teknologi
- Database gambar dari Kaggle (Potato Leaf Disease Dataset)
- Model machine learning dibangun menggunakan TensorFlow/Keras
- Aplikasi web interaktif menggunakan Streamlit
- Algoritma computer vision untuk preprocessing gambar

## Cara Penggunaan
1. Clone repository ini ke komputer Anda
2. Pastikan semua dependensi terinstal
3. Jalankan aplikasi web sesuai instruksi di bagian Instalasi
4. Ambil foto daun kentang menggunakan kamera atau unggah gambar dari perangkat
5. Sistem akan menganalisis dan menampilkan hasil prediksi penyakit

## Instalasi
```bash
# Clone repository
git clone https://github.com/username/deteksikentang.git
cd deteksikentang

# Instal dependensi
pip install -r requirements.txt

# Jalankan aplikasi
streamlit run main.py
```

## Fitur Utama
- Deteksi real-time menggunakan kamera
- Opsi untuk mengunggah gambar
- Visualisasi skor kepercayaan dengan gauge chart
- Interface yang intuitif dan mobile-friendly
- Informasi detail tentang setiap jenis penyakit

## Kontribusi
Kontribusi untuk pengembangan proyek ini sangat diterima. Silakan buat pull request atau laporkan issues.

## Lisensi
Sertakan nama asli jika mau memperbaiki kode ini

---

Dibuat oleh: Galuh Adi Insani