# Negative Comment Classifier

A machine learning application that classifies texts/comments based on their level of negativity 

Team members:
- 171111022	-- Suhendra
- 171111036	-- Nurami Nasrullah
- 171111069	-- Qori Hidayatulloh

## Langkah Pembuatan Program:

*  Data Collection

Aplikasi ini menggunakan pre-collected toxic comment dataset dari [Kaggle](https://www.kaggle.com/nichaoku/toxic-comment-merge-train-and-test-with-label).

* Data Preparation

Melakukan analisa awal dengan melihat isi baris dan kolom dari data untuk menentukan atribut mana yang cocok digunakan sebagai parameter. Kemudian membagi data ke dalam atribut dan label. Setelah itu membagi data ke dalam training dan test set.

* Choosing a Model

Menentukan algoritma apa yang akan digunakan untuk mengklasifikasi dataset. Aplikasi ini menggunakan algoritma SVM, Karena algoritma ini bisa mengklasifikasikan data dengan baik dan cepat.

* Training the Model

Melakukan training atau memberi pengetahuan terhadap algoritma sebelum bisa melakukan prediksi.

* Evaluating the Model

Menggunakan data sample modifikasi dari training set untuk melakukan evaluasi terhadap efektifitas dan efesiensi dari algoritma yang dipilih. Seperti tingkat akurasi dan lama pemrosesan. 

* Making Predictions

Membuat prediksi nyata dengan menggunakan sampel dari test set atau data yang tidak termasuk dalam training set. Input dari web kemudian dapat digunakan untuk melakukan prediksi data baru.

