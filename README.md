# AI Model Experiment & Evaluation

Eksperimen ini membandingkan dua pendekatan AI untuk melakukan sentiment analysis terhadap ulasan pelanggan pada platform e-commerce, yaitu model Machine Learning klasik menggunakan Scikit-learn dan Large Language Model (LLM) menggunakan Gemini AI API.

## 1. Problem Statement

### Objective

Tujuan dari eksperimen ini adalah membandingkan dua pendekatan AI untuk melakukan sentiment analysis terhadap ulasan pelanggan pada platform e-commerce.

Pendekatan yang dibandingkan adalah:

1. Model Machine Learning klasik menggunakan Scikit-learn.
2. Large Language Model (LLM) menggunakan Gemini AI API.

Eksperimen dilakukan untuk mengetahui performa dan trade-off dari kedua pendekatan sebelum menentukan technical approach yang sesuai untuk dikembangkan lebih lanjut.

### Target / Label

Target yang diprediksi adalah sentimen dari setiap ulasan pelanggan dengan dua kelas:

- **positif**
- **negatif**

Input model berupa teks ulasan pelanggan pada kolom `review_text`, sedangkan label target terdapat pada kolom `sentiment`.

### Batasan dan Asumsi

- Eksperimen menggunakan dataset ulasan pelanggan yang telah disediakan.
- Kedua pendekatan menggunakan test set yang sama agar perbandingan dapat dilakukan secara adil.
- Model klasik menggunakan preprocessing teks berbasis TF-IDF dan algoritma Logistic Regression.
- Pendekatan LLM menggunakan Gemini AI API dengan prompting tanpa proses training model.
- Evaluasi dilakukan menggunakan Accuracy, Precision, Recall, F1-Score, dan Confusion Matrix.
- Hasil eksperimen menggunakan 40 data test sehingga belum dapat langsung digeneralisasikan ke dataset yang lebih besar atau data produksi.

---

## 2. Dataset

Dataset yang digunakan merupakan dataset ulasan pelanggan e-commerce dengan struktur kolom:

| Kolom | Deskripsi |
|---|---|
| `review_id` | ID setiap ulasan |
| `product_name` | Nama produk |
| `review_text` | Teks ulasan pelanggan yang digunakan sebagai input model |
| `sentiment` | Label sentimen yang menjadi target prediksi |

Dataset terdiri dari **200 data**, dengan:

- **110 ulasan positif**
- **90 ulasan negatif**
- **Tidak terdapat missing value**

Dataset asli disimpan di:

`data/customer_reviews_sentiment.csv`

### Pembagian Data

Dataset dibagi menjadi:

- **80% training set:** 160 data
- **20% test set:** 40 data

Pembagian dilakukan menggunakan `train_test_split` dengan `random_state=42` dan `stratify=y`.

Test set yang sama digunakan oleh model Logistic Regression dan Gemini AI API agar hasil evaluasi kedua pendekatan dapat dibandingkan secara adil.

---

## 3. Eksperimen Pendekatan 1 — Model Klasik

Pendekatan pertama menggunakan model Machine Learning klasik dari Scikit-learn.

### Preprocessing

Teks ulasan diubah menjadi representasi numerik menggunakan **TF-IDF (Term Frequency-Inverse Document Frequency)**.

TF-IDF digunakan agar teks dapat direpresentasikan dalam bentuk fitur numerik yang dapat digunakan oleh algoritma Machine Learning.

### Model

Algoritma klasifikasi yang digunakan adalah **Logistic Regression**.

Alur eksperimen:

```text
Review Text
    ↓
TF-IDF
    ↓
Logistic Regression
    ↓
Prediksi Sentimen
```

Model dilatih menggunakan 160 data training dan kemudian digunakan untuk melakukan prediksi terhadap 40 data test.

---

## 4. Eksperimen Pendekatan 2 — Gemini AI API

Pendekatan kedua menggunakan Gemini AI API untuk melakukan klasifikasi sentimen tanpa proses training model.

### Prompt

Gemini diberikan instruksi untuk mengklasifikasikan setiap ulasan menjadi salah satu dari dua kelas:

- `positif`
- `negatif`

Model diminta memberikan output yang sederhana agar hasil prediksi dapat diproses dan dibandingkan dengan label sebenarnya.

### Parameter Gemini

Eksperimen menggunakan:

- **Temperature:** `0.2`

Nilai temperature yang relatif rendah dipilih agar output Gemini lebih konsisten dan fokus pada klasifikasi dua kelas sentimen, yaitu positif atau negatif, serta mengurangi variasi jawaban yang tidak diperlukan.

### Batch Inference

Karena terdapat batas penggunaan API, prediksi Gemini dilakukan secara bertahap menggunakan beberapa batch.

40 data test dibagi menjadi:

- Batch 1 → 10 data
- Batch 2 → 10 data
- Batch 3 → 10 data
- Batch 4 → 10 data

Hasil dari seluruh batch kemudian digabungkan menjadi 40 prediksi Gemini.

Alur eksperimen:

```text
Review Text
    ↓
Prompt
    ↓
Gemini AI API
    ↓
Prediksi Sentimen
```

---

## 5. Evaluasi dan Perbandingan

Kedua pendekatan dievaluasi menggunakan test set yang sama, yaitu 40 data.

Metrik yang digunakan:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

### 5.1 Accuracy

Accuracy menunjukkan proporsi seluruh prediksi yang berhasil diklasifikasikan dengan benar.

| Model | Accuracy |
|---|---:|
| Scikit-learn — Logistic Regression | 1.00 |
| Gemini AI API | 1.00 |

Kedua model menghasilkan Accuracy sebesar **1.00**, yang berarti seluruh 40 data test berhasil diklasifikasikan dengan benar pada eksperimen ini.

### 5.2 Precision

Precision menunjukkan seberapa tepat prediksi positif yang diberikan oleh model.

| Model | Precision |
|---|---:|
| Scikit-learn — Logistic Regression | 1.00 |
| Gemini AI API | 1.00 |

Kedua pendekatan memperoleh Precision sebesar **1.00** pada test set yang digunakan.

### 5.3 Recall

Recall menunjukkan kemampuan model dalam menemukan data yang termasuk ke dalam kelas tertentu.

| Model | Recall |
|---|---:|
| Scikit-learn — Logistic Regression | 1.00 |
| Gemini AI API | 1.00 |

Kedua pendekatan memperoleh Recall sebesar **1.00** pada eksperimen ini.

### 5.4 F1-Score

F1-Score merupakan metrik yang menggabungkan Precision dan Recall.

| Model | F1-Score |
|---|---:|
| Scikit-learn — Logistic Regression | 1.00 |
| Gemini AI API | 1.00 |

Kedua model memperoleh F1-Score sebesar **1.00**.

### 5.5 Confusion Matrix

Confusion matrix kedua model menghasilkan:

```text
[[18,  0],
 [ 0, 22]]
```

Interpretasinya:

- 18 data negatif berhasil diklasifikasikan sebagai negatif.
- 22 data positif berhasil diklasifikasikan sebagai positif.
- Tidak terdapat false positive.
- Tidak terdapat false negative.

Dengan demikian, seluruh data test berhasil diklasifikasikan dengan benar oleh kedua pendekatan pada eksperimen ini.

### 5.6 Tabel Perbandingan

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Scikit-learn — Logistic Regression | 1.00 | 1.00 | 1.00 | 1.00 |
| Gemini AI API | 1.00 | 1.00 | 1.00 | 1.00 |

Berdasarkan hasil pengujian pada 40 data test, kedua pendekatan menghasilkan nilai metrik yang sama.

---

## 6. Analisis Trade-off dan Limitation

### Performa

Berdasarkan hasil eksperimen pada 40 data test, model Logistic Regression berbasis TF-IDF dan Gemini AI API menghasilkan performa yang sama.

Kedua pendekatan memperoleh:

- Accuracy = 1.00
- Precision = 1.00
- Recall = 1.00
- F1-Score = 1.00

Confusion matrix kedua model juga menunjukkan bahwa 18 data negatif dan 22 data positif berhasil diklasifikasikan dengan benar.

Hasil tersebut menunjukkan bahwa pada dataset dan test set yang digunakan dalam eksperimen ini, kedua pendekatan mampu melakukan klasifikasi sentimen dengan hasil yang sama.

Namun, hasil tersebut belum dapat langsung digeneralisasikan ke dataset lain atau data produksi karena eksperimen hanya menggunakan 40 data test.

### Effort Implementasi

Model klasik membutuhkan beberapa tahap seperti preprocessing teks menggunakan TF-IDF, pembagian dataset, dan proses training Logistic Regression.

Setelah model selesai dilatih, prediksi dapat dilakukan secara lokal tanpa perlu mengirim data ke layanan eksternal.

Gemini AI API tidak membutuhkan proses training model. Implementasinya lebih sederhana dari sisi model karena klasifikasi dilakukan melalui prompt dan API.

Namun, diperlukan konfigurasi API dan pengelolaan request ke layanan Gemini.

### Kecepatan

Model Logistic Regression dapat melakukan prediksi secara lokal setelah model selesai dilatih sehingga tidak bergantung pada koneksi internet atau response time dari API.

Gemini membutuhkan request ke API untuk setiap proses inference. Pada eksperimen ini, pemanggilan API dilakukan secara bertahap menggunakan beberapa batch karena adanya batas penggunaan API.

Oleh karena itu, penggunaan Gemini perlu mempertimbangkan latency dan rate limit.

### Biaya

Model klasik tidak membutuhkan biaya API ketika digunakan untuk melakukan prediksi secara lokal setelah model dilatih. Biaya utamanya lebih berkaitan dengan resource komputasi yang digunakan untuk menjalankan model.

Gemini AI API berpotensi menimbulkan biaya penggunaan berdasarkan jumlah request dan penggunaan token, tergantung layanan dan konfigurasi API yang digunakan.

### Limitation Model Klasik

Keterbatasan model klasik adalah performanya bergantung pada dataset training dan representasi fitur yang digunakan.

Model juga perlu dilatih kembali apabila ingin disesuaikan dengan karakteristik data yang berbeda.

### Limitation Gemini

Keterbatasan Gemini dalam eksperimen ini adalah ketergantungan terhadap:

- API
- koneksi internet
- rate limit
- penggunaan token

Selama eksperimen, pemanggilan Gemini sempat mengalami error:

`429 RESOURCE_EXHAUSTED`

Oleh karena itu, inference 40 data perlu dilakukan menggunakan beberapa batch.

### Limitation Eksperimen

Kedua model memperoleh hasil sempurna pada test set yang hanya terdiri dari 40 data.

Oleh karena itu, hasil ini belum cukup untuk menyatakan bahwa kedua pendekatan akan memiliki performa yang sama pada dataset yang lebih besar atau data dunia nyata.

Pengujian dengan dataset yang lebih besar dan lebih beragam tetap diperlukan untuk memperoleh evaluasi yang lebih representatif.

---

## 7. Technical Recommendation

Berdasarkan hasil eksperimen, model Logistic Regression berbasis TF-IDF dan Gemini AI API menghasilkan performa yang sama pada test set yang digunakan.

Kedua pendekatan memperoleh:

- Accuracy = 1.00
- Precision = 1.00
- Recall = 1.00
- F1-Score = 1.00

Untuk use case klasifikasi sentimen sederhana pada dataset ini, pendekatan model klasik berbasis TF-IDF dan Logistic Regression direkomendasikan untuk dikembangkan lebih lanjut.

Pertimbangan utamanya adalah:

1. Model dapat dijalankan secara lokal setelah proses training.
2. Tidak bergantung pada API eksternal untuk inference.
3. Tidak menghadapi rate limit API ketika melakukan prediksi.
4. Tidak membutuhkan biaya API untuk inference lokal.
5. Model dapat digunakan kembali setelah proses training selesai.

Sementara itu, Gemini AI API tetap memiliki keunggulan dari sisi kemudahan implementasi karena tidak membutuhkan proses training model.

Pendekatan Gemini dapat menjadi alternatif ketika kebutuhan sistem membutuhkan kemampuan pemahaman bahasa yang lebih fleksibel atau ketika proses training dan pemeliharaan model lokal ingin diminimalkan.

Rekomendasi ini didasarkan pada hasil eksperimen saat ini. Karena pengujian hanya dilakukan pada 40 data test dan kedua pendekatan memperoleh hasil yang sama, pengujian dengan dataset yang lebih besar dan lebih beragam tetap diperlukan sebelum menentukan pendekatan untuk lingkungan produksi.

---

## 8. Project Structure

Struktur project:

```text
model-experiment-assignment/
├── data/
│   └── customer_reviews_sentiment.csv
├── notebook/
│   └── experiment_notebook.ipynb
├── documentation/
│   └── model_comparison_summary.png
├── README.md
└── requirements.txt
```

File `model_comparison_summary.png` bersifat opsional.

---

## 9. Requirements

Library yang digunakan dalam eksperimen:

```text
pandas
scikit-learn
google-genai
```

Pastikan Python dan pip sudah terinstall sebelum menjalankan project.

---

## 10. Cara Menjalankan Project

### 1. Clone Repository

Clone repository project ke komputer lokal:

```bash
git clone <URL_REPOSITORY>
```

Kemudian masuk ke folder project:

```bash
cd model-experiment-assignment
```

### 2. Install Dependencies

Install seluruh library yang dibutuhkan menggunakan:

```bash
pip install -r requirements.txt
```

### 3. Konfigurasi Gemini API Key

Gemini API membutuhkan API key.

API key disimpan menggunakan environment variable dan tidak ditulis langsung di notebook.

Pada Windows PowerShell:

```powershell
$env:GEMINI_API_KEY="YOUR_API_KEY"
```

Ganti `YOUR_API_KEY` dengan API key Gemini milik sendiri.

### 4. Jalankan Notebook

Buka project menggunakan VS Code dan buka:

```text
notebook/experiment_notebook.ipynb
```

Pastikan kernel Python yang digunakan sudah memiliki dependencies yang diperlukan.

Kemudian jalankan cell notebook sesuai urutan dari atas ke bawah.

Notebook berisi:

1. Load dataset
2. Train/test split
3. TF-IDF preprocessing
4. Training Logistic Regression
5. Gemini API inference
6. Evaluation
7. Comparison
8. Trade-off analysis
9. Technical recommendation

---

## 11. Kesimpulan

Eksperimen membandingkan dua pendekatan sentiment analysis, yaitu Logistic Regression berbasis TF-IDF dan Gemini AI API.

Pada test set yang terdiri dari 40 data, kedua pendekatan menghasilkan:

```text
Accuracy  = 1.00
Precision = 1.00
Recall    = 1.00
F1-Score  = 1.00
```

Meskipun hasil performanya sama pada eksperimen ini, terdapat perbedaan dari sisi implementasi, inference, ketergantungan terhadap API, rate limit, dan potensi biaya.

Berdasarkan pertimbangan tersebut, model Logistic Regression berbasis TF-IDF dipilih sebagai technical approach yang direkomendasikan untuk use case klasifikasi sentimen sederhana pada dataset eksperimen ini.

Namun, pengujian lebih lanjut menggunakan dataset yang lebih besar dan lebih beragam diperlukan sebelum digunakan pada lingkungan produksi.