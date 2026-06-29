# Klasifikasi Tingkat Keparahan Alzheimer menggunakan CNN & Transfer Learning

Repositori ini berisi implementasi pipeline Machine Learning untuk mendeteksi tingkat keparahan penyakit Alzheimer (Demensia) dari citra MRI, dikembangkan sebagai pemenuhan Tugas Besar Mata Kuliah Machine Learning.

## Anggota Kelompok Kapal Karam
1. **ANDI MUHAMMAD ABID JAYA (103132400014)** (Role: Data Specialist) - Penyusunan DataLoader, augmentasi, dan preprocessing citra.
2. **Ahnaf Ariacheda (103132400037)** (Role: ML Engineer) - Pengembangan arsitektur PyTorch dan script training.
3. **Ardian Nafis Samudra (103132400012)** (Role: ML Analyst & Ops) - Evaluasi metrik performa, Git ops, dan penyusunan laporan akhir.

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
* **Lisensi Dataset:** 

[Kaggle Data License & Terms](https://www.kaggle.com/terms) (Diperuntukkan murni untuk riset akademik dan non-komersial)

[CC0 1.0 Universal (Public Domain Dedication)](https://creativecommons.org/publicdomain/zero/1.0/) - Dataset ini berstatus bebas hak cipta (*No Rights Reserved*). Seluruh citra di dalamnya telah didedikasikan untuk domain publik, sehingga sepenuhnya legal dan aman untuk diunduh, dimodifikasi, dan digunakan untuk keperluan riset akademik ini tanpa batasan hak cipta.

---

## 3. Tahapan Preprocessing
Sesuai dengan standar *best practice* dan rekomendasi dari uploader Kaggle, pipeline pra-pemrosesan diatur dalam `data_loader.py` dengan rincian:
1. **Resizing:** Mengubah resolusi seluruh citra menjadi ukuran statis `224x224` piksel.
2. **Normalisasi ImageNet:** Mengonversi tensor citra ke rentang `[0, 1]` dan menormalisasi menggunakan *mean* `[0.485, 0.456, 0.406]` dan *std* `[0.229, 0.224, 0.225]`.
3. **Isolasi Augmentasi (Steril):** Augmentasi citra (*Random Horizontal Flip*, *Random Rotation 15°*, dan *Color Jitter*) **hanya** diaplikasikan pada data *training* untuk mencegah *Data Augmentation Leakage*. Data *testing* dibiarkan murni untuk evaluasi objektif.

---

## 4. Hasil Eksperimen dan Kesimpulan

Sebagai tahap awal (*baseline*), kami melakukan eksperimen menggunakan arsitektur **CNN Dasar** konvensional dengan 2 lapis konvolusi sebagai acuan performa. Selain itu, kami juga menerapkan pendekatan *Transfer Learning* dengan **MobileNetV2** untuk mengoptimalkan klasifikasi pada fitur citra medis yang kompleks.

### Tabel Perbandingan Performa Model
Berikut adalah ringkasan hasil evaluasi akurasi dari kedua model yang telah diuji:

| Model | Durasi Training | Epoch | Overall Accuracy |
| :--- | :--- | :--- | :--- |
| **CNN Dasar** | 40 Epoch | 40 | 77.71% |
| **MobileNetV2** | 15 Epoch | 15 | 91.48% |

### Detail Metrik Evaluasi (F1-Score)
Perbandingan kemampuan model dalam mengklasifikasikan setiap kategori penyakit dapat dilihat pada grafik berikut:


<img width="1089" height="580" alt="image" src="https://github.com/user-attachments/assets/3a5c3f6b-e80d-43cc-a3e9-6c07765608ca" />


### Perbandingan Performa Model (Studi Komparatif)

Sebagai studi komparatif, kami melakukan eksperimen menggunakan arsitektur **CNN Dasar** konvensional (2 lapis konvolusi) sebagai acuan awal. Setelah proses *training* selama 40 *epoch*, performa model tersebut mencapai akurasi **77.71%**. Hasil ini menunjukkan bahwa arsitektur CNN dasar kurang optimal untuk mengekstrak fitur medis yang rumit pada citra MRI. Oleh karena itu, kami beralih menggunakan pendekatan *Transfer Learning* dengan **MobileNetV2** yang terbukti jauh lebih efisien dan superior, sukses mencapai akurasi **91.48%** hanya dalam 15 *epoch*.

> **Catatan:** Log eksekusi dan metrik perbandingan detail dari kedua model ini didokumentasikan pada file `Komparasi_CNN_&_MobileNetV2.ipynb` yang tersedia di repositori ini.

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
* **Materi Akademik Utama:** Kelas Mata Kuliah Machine Learning - PS1SD-05-REG01 (Computer Vision / Live Code CNN).
* **Referensi Base Code (Kaggle Notebooks):** Pendekatan klasifikasi, arsitektur, dan tuning diadaptasi dari [Alzheimer MRI Code Documentation](https://www.kaggle.com/datasets/mohiburrahmanrifat/alzheimer/code) yang disertakan bersama dataset.
* **Arsitektur Model:** Perancangan model menggunakan struktur *Sequential CNN* dan *Transfer Learning* merujuk pada [PyTorch Official Documentation](https://pytorch.org/docs/stable/nn.html).
* **PyTorch Documentation:** Dokumentasi resmi untuk [Torchvision Models (MobileNetV2)](https://pytorch.org/vision/stable/models.html) dan [Data Loaders](https://pytorch.org/docs/stable/data.html).
* **Scikit-Learn Documentation:** Referensi modul [Classification Report & Metrics](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics).
