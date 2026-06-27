# Tugas-Besar_Machine-Learning_PS1SD-05-REG01

# Prediksi Penyakit Alzheimer Menggunakan Machine Learning

## Nama Anggota Kelompok (Pamplona)
1. Ahnaf (Data Engineer / Analyst)
2. Ara (Machine Learning Engineer)
3. [Nama Anggota 3] (Evaluator & Documenter)

## Deskripsi Permasalahan
Penyakit Alzheimer adalah gangguan neurologis progresif. Proyek ini bertujuan untuk menyelesaikan permasalahan nyata di bidang medis, yaitu melakukan klasifikasi dan deteksi dini status demensia pasien berdasarkan data rekam klinis.

## Sumber dan Deskripsi Dataset
Dataset yang digunakan berasal dari Kaggle: [Alzheimer Disease Dataset](https://www.kaggle.com/datasets/mohiburrahmanrifat/alzheimer/data). Dataset (`Alzheimer_Dataset_Details.csv`) berisi data klinis pasien berformat tabular yang mencakup usia, tingkat pendidikan, hasil tes kognitif, dan status diagnosis.

## Tahapan Preprocessing
1. Mengimpor data menggunakan library Pandas.
2. Mengisi nilai yang kosong (*missing values*) menggunakan nilai median.
3. Melakukan *encoding* untuk mengubah fitur kategorikal/teks menjadi numerik menggunakan `LabelEncoder` dari Scikit-Learn.
4. Membagi data menjadi 80% data latih dan 20% data uji.

## Metode yang Digunakan
Proyek ini mengimplementasikan dan membandingkan tiga algoritma Machine Learning:
1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)

*(Referensi resmi penggunaan modul disertakan di dalam komentar source code).*

## Cara Menjalankan Program
1. Pastikan Anda memiliki Python 3.x dan library pendukung (Pandas, Scikit-Learn) yang telah terinstal.
2. Unduh atau *clone* repositori ini.
3. Pastikan file `Alzheimer_Dataset_Details.csv` berada di direktori yang sama.
4. Jalankan perintah berikut di terminal:
   `python main.py`

## Hasil Eksperimen dan Evaluasi
Proses evaluasi menggunakan metrik akurasi, presisi, dan *recall* untuk melihat model mana yang paling optimal dalam mengklasifikasikan status Alzheimer pasien. *(Catatan: Tambahkan hasil akhir di sini setelah program dijalankan).*

## Kesimpulan
Membandingkan beberapa algoritma pada data medis menunjukkan bahwa (*Catatan: Isi dengan algoritma pemenang*) memberikan hasil terbaik. Penggunaan Machine Learning terbukti dapat membantu klasifikasi awal berdasarkan rekam medis klinis pasien.
