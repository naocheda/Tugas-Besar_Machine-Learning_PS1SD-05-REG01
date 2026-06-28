# Klasifikasi Tingkat Keparahan Alzheimer menggunakan CNN & Transfer Learning

Repositori ini berisi implementasi pipeline Machine Learning untuk mendeteksi tingkat keparahan penyakit Alzheimer (Demensia) dari citra MRI, dikembangkan sebagai pemenuhan Tugas Besar Mata Kuliah Machine Learning.

## Anggota Kelompok
1. **Ahnaf** (Role: ML Engineer) - Pengembangan arsitektur PyTorch dan script training.
2. **Abid** (Role: Data Specialist) - Penyusunan DataLoader, augmentasi, dan preprocessing citra.
3. **Ardian** (Role: ML Analyst & Ops) - Evaluasi metrik performa, Git ops, dan penyusunan laporan akhir.

---

## 1. Deskripsi Permasalahan
Penyakit Alzheimer merupakan gangguan neurodegeneratif progresif. Proyek ini bertujuan untuk membangun model Computer Vision yang mampu secara otomatis mengklasifikasikan citra MRI otak ke dalam tiga kelas keparahan:
- `Non_Demented` (Sehat)
- `Very_Mild_Demented` (Gejala Awal/Sangat Ringan)
- `Mild_Demented` (Gejala Ringan)

Pendekatan eksperimen difokuskan pada perbandingan antara arsitektur Convolutional Neural Network (CNN) dasar dan pemanfaatan Transfer Learning menggunakan **MobileNetV2**.

---

## 2. Sumber Dataset & Lisensi
Dataset menggunakan citra otak MRI format `.jpg` (total 6.336 citra) dengan metadata berformat CSV.
* **Tautan Dataset Utama:** [Alzheimer MRI Dataset on Kaggle](https://www.kaggle.com/datasets/mohiburrahmanrifat/alzheimer/data)
* **Lisensi Dataset:** [Kaggle Data License & Terms](https://www.kaggle.com/terms) (Diperuntukkan murni untuk riset akademik dan non-komersial).

---

## 3. Tahapan Preprocessing
Sesuai dengan standar *best practice* dan rekomendasi dari uploader Kaggle, pipeline pra-pemrosesan diatur dalam `data_loader.py` dengan rincian:
1. **Resizing:** Mengubah resolusi seluruh citra menjadi ukuran statis `224x224` piksel.
2. **Normalisasi ImageNet:** Mengonversi tensor citra ke rentang `[0, 1]` dan menormalisasi menggunakan *mean* `[0.485, 0.456, 0.406]` dan *std* `[0.229, 0.224, 0.225]`.
3. **Isolasi Augmentasi (Steril):** Augmentasi citra (*Random Horizontal Flip*, *Random Rotation 15°*, dan *Color Jitter*) **hanya** diaplikasikan pada data *training* untuk mencegah *Data Augmentation Leakage*. Data *testing* dibiarkan murni untuk evaluasi objektif.

---

## 4. Hasil Eksperimen dan Kesimpulan
Pengujian model menggunakan arsitektur **MobileNetV2** selama 15 epoch memberikan hasil yang sangat konvergen dan superior. Model dievaluasi menggunakan 951 citra pengujian yang tidak pernah dilihat sebelumnya (unseen data), dengan metrik akhir:

* **Overall Accuracy:** 91.48%
* **Precision Rata-rata (Weighted Avg):** 92.00%
* **Recall Rata-rata (Weighted Avg):** 91.00%
* **F1-Score Rata-rata (Weighted Avg):** 92.00%

**Kesimpulan Klinis:**
Model menunjukkan sensitivitas tinggi dengan **Recall sebesar 0.93** pada kelas kritis (`Very_Mild_Demented`). Dalam konteks skrining medis, meminimalkan *False Negative* pada fase awal penyakit adalah prioritas utama, dan model telah berhasil mencapainya. Penggunaan MobileNetV2 juga membuktikan ketahanan model terhadap imbalanced data (jumlah kelas yang tidak seimbang) dengan menjaga F1-Score tetap stabil di atas angka 0.90 di seluruh kelas.

---

## 5. Ancaman Validitas (Threats to Validity) & Limitasi
Meskipun model mencapai akurasi **91.48%**, penting untuk mencatat bahwa dataset publik ini berpotensi memiliki *Patient-Level Data Leakage*. Pengacakan data (splitting) ke folder `train` dan `test` yang dilakukan oleh penyedia dataset mungkin memotong *slice* MRI dari pasien yang sama. Akibatnya, model berisiko mengenali "anatomi spesifik pasien" alih-alih murni fitur patologis demensia. Untuk implementasi medis sesungguhnya, pengujian dengan metode *Leave-One-Patient-Out Cross Validation* mutlak diperlukan.

---

## 6. Cara Menjalankan Program
Untuk mencegah kelebihan kapasitas (limit size) pada repositori GitHub, citra raw MRI tidak diikutsertakan. Ikuti instruksi ini untuk melakukan reproduksi hasil (lokal/Google Colab):
1. Lakukan `git clone` pada repositori ini.
2. Unduh file `archive.zip` dari [tautan Kaggle ini](https://www.kaggle.com/datasets/mohiburrahmanrifat/alzheimer/data).
3. Ekstrak file ZIP tersebut langsung ke dalam *root directory* repositori ini (pastikan struktur folder `train` dan `test` sejajar dengan file `train.py`).
4. Instalasi dependensi: `pip install -r requirements.txt`
5. Jalankan `python train.py` untuk memulai pelatihan model.
6. Jalankan `python evaluation.py` untuk mencetak metrik performa akhir.

---

## 7. Referensi Pustaka dan Kode
Repositori ini dikembangkan dengan merujuk pada dokumentasi dan literatur berikut:
* **Materi Akademik Utama:** Modul Pembelajaran Sains Data (Computer Vision / Live Code CNN).
* **Referensi Base Code (Kaggle Notebooks):** Pendekatan klasifikasi, arsitektur, dan tuning diadaptasi dari [Alzheimer MRI Code Documentation](https://www.kaggle.com/datasets/mohiburrahmanrifat/alzheimer/code) yang disertakan bersama dataset.
* **PyTorch Documentation:** Dokumentasi resmi untuk [Torchvision Models (MobileNetV2)](https://pytorch.org/vision/stable/models.html) dan [Data Loaders](https://pytorch.org/docs/stable/data.html).
* **Scikit-Learn Documentation:** Referensi modul [Classification Report & Metrics](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics).
