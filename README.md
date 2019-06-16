# Negative Comment Classifier
A machine learning application that classifies texts/comments based on their level of negativity 

Team members:
- 171111022	-- Suhendra
- 171111036	-- Nurami Nasrullah
- 171111069	-- Qori Hidayatulloh

## Langkah Pembuatan Program:

1. Data Collection

Program ini menggunakan pre-collected toxic comment dataset

2. Data Preparation

Melakukan analisa awal dengan melihat isi baris dan kolom dari data untuk menentukan atribut mana yang cocok digunakan sebagai parameter. Kemudian membagi data ke dalam atribut dan label. Setelah itu membagi data ke dalam training dan test set.

3. Choose a Model

Menentukan algoritma apa yang akan digunakan untuk mengklasifikasi dataset. Program ini menggunakan algoritma SVM, Karena algoritma ini bisa mengklasifikasikan data dengan baik dan cepat.

4. Train the Model

Melakukan training atau memberi pengetahuan terhadap algoritma sebelum bisa melakukan prediksi.

5. Evaluate the Model

Menggunakan data sample modifikasi dari training set untuk melakukan evaluasi terhadap efektifitas dan efesiensi dari algoritma yang dipilih. Seperti tingkat akurasi dan lama pemrosesan. 

6. Make Predictions

Membuat prediksi nyata dengan menggunakan sampel dari test set atau data yang tidak termasuk dalam training set.

## Library Yang Dibutuhkan:

1. pandas			      : mengolah dataset menjadi table dataframe yang bisa dibaca
2. TfidfVectorizer	: mengubah dataset menjadi vector agar bisa digunakan sebagai input untuk proses prediksi.
3. train_test_split	: membagi dataset kedalam Training Set dan Test Set serta meminimalisir overfitting atau underfitting dengan mengacak dataset.
4. LinearSVC		    : algoritma untuk mengklasifikasi dan memprediksi data. LinearSVC dipilih karena proses eksekusinya lebih cepat dan mampu memisahkan data dengan baik 
5. accuracy_score	  : menghitung akurasi dari hasil prediksi
